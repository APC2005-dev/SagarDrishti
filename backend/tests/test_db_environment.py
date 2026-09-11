"""Environmental pipeline against PostGIS with synthetic (offline) providers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.session import get_async_session
from app.main import app
from app.models.environment import EnvCacheEntry, EnvIngestionRun, ObservationEnvironment
from app.models.ml import ForecastSet, ModelStatusEvent, ModelVersion
from app.services.environment_service import EnvironmentService
from app.services.forecast_service import ForecastService
from app.services.ingestion_service import IngestionService
from app.services.model_registry import ModelRegistry
from app.services.retraining_service import RetrainingService
from ml.environment.types import EnvGroup, ProviderRole
from ml.tests.test_environment import FakeProvider
from tests.helpers import add_historical_track, csv_rows, document, write_tiny_base

pytestmark = [pytest.mark.db, pytest.mark.tf]

A81_0910 = "A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026"
A81_0917 = "A81,28,25,-57.50,-47.00,518.07,391.20,1341.79,09/17/2026"


def fake_providers(fail: bool = False) -> dict[ProviderRole, dict[EnvGroup, FakeProvider]]:
    # A failing source gets its own name: the tile cache (correctly) serves any
    # previously fetched data for a source, so reusing a name would hide the failure.
    def trio(prefix: str) -> dict[EnvGroup, FakeProvider]:
        prefix = f"down_{prefix}" if fail else prefix
        return {
            EnvGroup.WIND: FakeProvider(EnvGroup.WIND, f"{prefix}_wind", fail=fail),
            EnvGroup.CURRENT: FakeProvider(EnvGroup.CURRENT, f"{prefix}_current", fail=fail),
            EnvGroup.SEA_ICE: FakeProvider(EnvGroup.SEA_ICE, f"{prefix}_ice", fail=fail),
        }

    return {ProviderRole.OPERATIONAL: trio("op"), ProviderRole.HISTORICAL: trio("hist")}


@pytest.fixture
def env_settings(settings):  # type: ignore[no-untyped-def]
    return settings.model_copy(update={
        "env_enabled": True, "retrain_epochs": 1, "env_training_max_historical_samples": 300,
        "env_training_max_test_samples": 200, "env_candidate_schemas": ["trajectory_v1", "traj_wind_v1"],
    })


@pytest.fixture
def pipeline(db, settings):  # type: ignore[no-untyped-def]
    write_tiny_base(Path(settings.models_dir))
    add_historical_track(db, "A81", date(2026, 8, 1), 30)
    ModelRegistry(db, settings).ensure_v1_bootstrap()
    db.commit()
    return db


def test_alignment_stores_provenance_and_indexes_cache(pipeline, env_settings, fixture_csv) -> None:  # type: ignore[no-untyped-def]
    db = pipeline
    IngestionService(db, env_settings).run(document=document(fixture_csv))
    env = EnvironmentService(db, env_settings, providers=fake_providers())
    out = env.align_observations(as_of=date(2026, 9, 11))
    assert out["observations"] == 33 and out["aligned"] == 99
    row = db.execute(select(ObservationEnvironment).where(ObservationEnvironment.iceberg_id == "A81",
                                                          ObservationEnvironment.group == "wind")).scalar_one()
    assert row.valid_date == date(2026, 9, 10) and row.as_of == date(2026, 9, 11) and row.staleness_days == 0
    assert row.values["wind_u"] is not None and row.interpolation == "bilinear_nan_aware_daily"
    assert row.provider == "op_wind" and row.units == {"wind_u": "1", "wind_v": "1"}
    assert db.scalar(select(func.count()).select_from(EnvCacheEntry)) > 0
    assert db.scalar(select(func.count()).select_from(EnvIngestionRun).where(EnvIngestionRun.kind == "tile_fetch")) > 0
    assert env.align_observations(as_of=date(2026, 9, 11))["aligned"] == 0  # idempotent
    # append-only provenance
    with pytest.raises(DBAPIError):
        db.execute(text("UPDATE environmental.observation_features SET missing = true"))
    db.rollback()


def test_alignment_skipped_when_disabled_or_unconfigured(pipeline, settings, fixture_csv) -> None:  # type: ignore[no-untyped-def]
    IngestionService(pipeline, settings).run(document=document(fixture_csv))
    assert EnvironmentService(pipeline, settings).align_observations()["status"] == "skipped"  # ENV_ENABLED=false in tests
    on = settings.model_copy(update={"env_enabled": True})
    assert EnvironmentService(pipeline, on).align_observations()["status"] == "skipped"  # no credentials
    assert pipeline.scalar(select(func.count()).select_from(ObservationEnvironment)) == 0


def _deploy_env_model(db, settings) -> str:  # type: ignore[no-untyped-def]
    """Train a tiny traj_wind_v1 model on synthetic data and deploy it as v2 (lineage parent v1)."""
    import pandas as pd

    from ml.environment.alignment import attach_environment
    from ml.environment.cache import EnvironmentalCache
    from ml.environment.feature_builder import EnvironmentalFeatureBuilder
    from ml.features.coordinate_transform import latlon_to_polar_m
    from ml.features.schemas import get_schema
    from ml.models.artifact_store import write_version
    from ml.training.dataset_builder import build_historical_sequences
    from ml.training.environmental import train_schema_candidate
    from ml.training.trainer import TrainingConfig

    days = 50
    lat = -64 - np.arange(days) * 0.03
    lon = 40 + np.arange(days) * 0.02
    x, y = latlon_to_polar_m(lat, lon)
    tracks = pd.DataFrame({"iceberg_id": "S", "date": pd.date_range("2024-02-01", periods=days), "latitude": lat,
                           "longitude": lon, "x_m": x, "y_m": y})
    ds = build_historical_sequences(tracks, with_entries=True)
    builder = EnvironmentalFeatureBuilder({EnvGroup.WIND: FakeProvider()}, EnvironmentalCache(Path(settings.env_cache_dir) / "train"))
    full, _ = attach_environment(ds, builder, ("wind_u", "wind_v"))
    schema = get_schema("traj_wind_v1")
    oc = train_schema_candidate(schema, full, full, TrainingConfig(epochs=1, batch_size=8), "v2", min_samples=5)
    meta = {"version": "v2", "parent_version": "v1", "architecture": "GRU", "architecture_version": oc.architecture_version,
            "input_sequence_length": 14, "input_semantics": "chronological_observation_entries", "forecast_horizon_days": 7,
            "feature_schema_version": schema.version, "feature_names": list(schema.features), "adapter_strategy": "bootstrap_v1",
            "artifact_origin": "test", "environmental_data_sources": {"wind": {"operational": {"dataset_id": "op_wind_ds"}}},
            "environmental_data_cutoff": "2026-09-10",
            "metrics": {"by_horizon": {str(h): {"n": 10, "mae_km": float(h), "p90_km": 2.0 * h} for h in range(1, 8)}}}
    target = Path(settings.models_dir)
    written = write_version(target, "v2", oc.bundle.model, oc.bundle.feature_scaler, oc.bundle.target_scaler, meta,
                            feature_names=schema.features)
    reg = ModelRegistry(db, settings)
    mv = reg._row_from_metadata("v2", target / "v2", json.loads(written.metadata_path.read_text()), written.checksums, "candidate")
    reg.record_metrics("v2", "operational_test", meta["metrics"]["by_horizon"], "test")
    reg.transition(mv, "validated", "test")
    reg.promote(mv, "test deployment")
    db.commit()
    return "v2"


def test_environmental_champion_forecast_has_provenance(pipeline, env_settings) -> None:  # type: ignore[no-untyped-def]
    db = pipeline
    _deploy_env_model(db, env_settings)
    v2 = db.execute(select(ModelVersion).where(ModelVersion.version == "v2")).scalar_one()
    assert v2.model_type == "environmental" and v2.feature_schema_version == "traj_wind_v1" and v2.parent_version == "v1"
    IngestionService(db, env_settings).run(document=document(csv_rows(A81_0910)))
    env = EnvironmentService(db, env_settings, providers=fake_providers())
    out = ForecastService(db, env_settings, environment=env).run(trigger="test")
    assert out.created_sets == 1 and out.fallback_sets == 0
    fs = db.execute(select(ForecastSet)).scalar_one()
    assert fs.model_version == "v2" and fs.feature_schema_version == "traj_wind_v1"
    assert fs.environment_as_of == datetime.now(UTC).date() and fs.fallback is None
    assert np.asarray(fs.input_features).shape == (14, 8)
    anchor_env = fs.input_entries[-1]["environment"]["wind"]
    assert anchor_env["provider"] == "op_wind" and anchor_env["valid_date"] <= str(fs.environment_as_of)
    assert all(e["wind"]["valid_date"] <= e["wind"]["as_of"] for e in fs.environment)  # nothing after T


def test_missing_environment_falls_back_to_trajectory_model(pipeline, env_settings) -> None:  # type: ignore[no-untyped-def]
    db = pipeline
    _deploy_env_model(db, env_settings)
    IngestionService(db, env_settings).run(document=document(csv_rows(A81_0910)))
    env = EnvironmentService(db, env_settings, providers=fake_providers(fail=True))
    out = ForecastService(db, env_settings, environment=env).run(trigger="test")
    assert out.fallback_sets == 1
    fs = db.execute(select(ForecastSet)).scalar_one()
    assert fs.model_version == "v1" and fs.feature_schema_version == "trajectory_v1"
    assert fs.fallback["champion"] == "v2" and fs.fallback["missing_count"] > 0
    assert np.asarray(fs.input_features).shape == (14, 6)


def test_feature_schema_experiment(pipeline, env_settings) -> None:  # type: ignore[no-untyped-def]
    db = pipeline
    add_historical_track(db, "H1", date(2017, 1, 1), 2600, lat0=-64.0, lon0=30.0)
    IngestionService(db, env_settings).run(document=document(csv_rows(A81_0910)))
    IngestionService(db, env_settings).run(document=document(csv_rows(A81_0917), datetime(2026, 9, 18, tzinfo=UTC)))
    env = EnvironmentService(db, env_settings, providers=fake_providers())
    run = RetrainingService(db, env_settings, environment=env).run(trigger="test", force=True)
    assert run.status in ("promoted", "rejected"), run.failure_reason
    exp = run.experiment
    assert [c["schema"] for c in exp["candidates"]] == ["trajectory_v1", "traj_wind_v1"]
    rows = {m.version: m for m in db.execute(select(ModelVersion)).scalars()}
    v2, v3 = rows["v2"], rows["v3"]
    assert (v2.feature_schema_version, v2.model_type) == ("trajectory_v1", "trajectory")
    assert (v3.feature_schema_version, v3.model_type) == ("traj_wind_v1", "environmental")
    assert v3.environmental_data_sources["wind"]["historical"]["name"] == "hist_wind" and v3.parent_version == "v1"
    selected = run.decision["selected_version"]
    other = "v3" if selected == "v2" else "v2"
    assert rows[other].status == "rejected" and rows[other].status_reason.startswith("not selected")
    assert rows[selected].status == ("deployed" if run.status == "promoted" else "rejected")
    assert "vs base" in " ".join(run.decision["reasons"])
    assert "wind" in run.metrics["test"]["effects"]
    assert set(run.metrics["test"]["models"]) >= {"base", "champion:v1", "trajectory_v1", "traj_wind_v1"}
    assert db.scalar(select(func.count()).select_from(ModelVersion).where(ModelVersion.status == "deployed")) == 1
    meta = json.loads((Path(env_settings.models_dir) / "v3" / "metadata.json").read_text())
    assert meta["feature_schema_version"] == "traj_wind_v1" and meta["environmental_policy"]["as_of_rule"]
    events = [e.to_status for e in db.execute(select(ModelStatusEvent).where(ModelStatusEvent.model_version == "v3")).scalars()]
    assert events[:2] == ["candidate", "validated"]


@pytest.fixture
async def client(settings) -> AsyncIterator[httpx.AsyncClient]:  # type: ignore[no-untyped-def]
    engine = create_async_engine(settings.async_database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def override() -> AsyncIterator[AsyncSession]:
        async with factory() as s:
            yield s

    app.dependency_overrides[get_async_session] = override
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


async def test_environment_api(client, pipeline, env_settings, fixture_csv) -> None:  # type: ignore[no-untyped-def]
    IngestionService(pipeline, env_settings).run(document=document(fixture_csv))
    r = await client.get("/api/v1/icebergs/A81/environment")
    assert r.status_code == 200 and r.json()["aligned"] is False and "ENV_ENABLED" in r.json()["reason"]
    EnvironmentService(pipeline, env_settings, providers=fake_providers()).align_observations(as_of=date(2026, 9, 11))
    body = (await client.get("/api/v1/icebergs/A81/environment")).json()
    assert body["aligned"] and {g["group"] for g in body["groups"]} == {"wind", "current", "sea_ice"}
    wind = next(g for g in body["groups"] if g["group"] == "wind")
    assert wind["vector"]["directionConvention"] == "from" and wind["vector"]["speedMS"] > 0
    status = (await client.get("/api/v1/environment/status")).json()
    assert status["enabled"] is False and {s["group"] for s in status["sources"]} == {"wind", "current", "sea_ice"}
    assert all(not s["configured"] for s in status["sources"])
    assert any(s["datasetId"] == "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m" for s in status["sources"])
    assert (await client.get("/api/v1/environment/field", params={"group": "wind"})).status_code == 404
    runs = (await client.get("/api/v1/environment/runs")).json()
    assert any(r["kind"] == "alignment" for r in runs)
    feeds = (await client.get("/api/v1/feeds")).json()
    env_feeds = [f for f in feeds if f["category"] == "environmental"]
    assert env_feeds and all(f["state"] == "UNKNOWN" and not f["configured"] for f in env_feeds)
