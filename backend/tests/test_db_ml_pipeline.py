"""Registry, forecast generation, evaluation and retraining against PostGIS + TensorFlow."""

from __future__ import annotations

import os
import stat
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.ml import Forecast, ForecastEvaluation, ForecastSet, ModelStatusEvent, ModelVersion
from app.services.evaluation_service import EvaluationService
from app.services.forecast_service import ForecastService
from app.services.ingestion_service import IngestionService
from app.services.model_registry import ModelRegistry, RegistryError
from app.services.retraining_service import RetrainingService
from ml.features.coordinate_transform import haversine_km
from tests.helpers import add_historical_track, base_p90, csv_rows, document, write_tiny_base

pytestmark = [pytest.mark.db, pytest.mark.tf]

A81_0910 = "A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026"
A81_0917 = "A81,28,25,-57.50,-47.00,518.07,391.20,1341.79,09/17/2026"
D37_0910 = "D37,30,7,-69.21,36.36,184.47,139.29,477.77,09/10/2026"


@pytest.fixture
def pipeline(db, settings):  # type: ignore[no-untyped-def]
    write_tiny_base(Path(settings.models_dir))
    add_historical_track(db, "A81", date(2026, 8, 1), 30)
    ModelRegistry(db, settings).ensure_v1_bootstrap()
    db.commit()
    return db


def test_model_unavailable_is_reported_not_faked(db, settings) -> None:  # type: ignore[no-untyped-def]
    out = ForecastService(db, settings).run()
    assert out.status == "skipped" and out.error == "no deployed model version"
    assert db.scalar(select(func.count()).select_from(Forecast)) == 0


def test_bootstrap_registers_base_and_deploys_v1(pipeline, settings) -> None:  # type: ignore[no-untyped-def]
    reg = ModelRegistry(pipeline, settings)
    base, v1 = reg.get("base"), reg.get("v1")
    assert base.status == "validated" and v1.status == "deployed" and v1.parent_version == "base"
    assert v1.adapter_strategy == "bootstrap_v1" and v1.input_semantics == "chronological_observation_entries"
    assert v1.artifact_sha256["model.keras"] == base.artifact_sha256["global_gru_trajectory_model.keras"]
    assert not os.stat(Path(settings.models_dir) / "v1" / "model.keras").st_mode & stat.S_IWUSR
    assert reg.risk_radii("v1") == base_p90()
    events = [e.to_status for e in pipeline.execute(select(ModelStatusEvent).where(ModelStatusEvent.model_version == "v1")).scalars()]
    assert events == ["candidate", "validated", "deployed"]
    assert reg.ensure_v1_bootstrap().version == "v1"  # idempotent


def test_forecast_then_evaluate_full_lineage(pipeline, settings) -> None:  # type: ignore[no-untyped-def]
    db = pipeline
    IngestionService(db, settings).run(document=document(csv_rows(A81_0910, D37_0910)))
    out = ForecastService(db, settings).run(trigger="test")
    assert out.created_sets == 1 and out.skipped == {"insufficient_history": 1}  # D37 has no history: skipped, not invented

    fs = db.execute(select(ForecastSet)).scalar_one()
    assert fs.model_version == "v1" and fs.latest_observation_date == date(2026, 9, 10)
    assert len(fs.input_observation_ids) == 14 and fs.input_entries[-1]["provenance"] == "official_usnic"
    assert [e["provenance"] for e in fs.input_entries[:13]] == ["historical_training_dataset"] * 13
    assert fs.diagnostics["daily_cadence"] is False
    points = db.execute(select(Forecast).order_by(Forecast.forecast_horizon_days)).scalars().all()
    assert [p.forecast_horizon_days for p in points] == list(range(1, 8))
    assert points[6].forecast_date == date(2026, 9, 17) and points[0].risk_radius_km_p90 == base_p90()[1]
    assert all(p.provenance == "predicted" for p in points)

    # idempotent: same anchor + same model -> nothing new
    assert ForecastService(db, settings).run().already_forecast == 1

    later = IngestionService(db, settings).run(document=document(csv_rows(A81_0917), datetime(2026, 9, 18, tzinfo=UTC)))
    ev = EvaluationService(db).evaluate(later.changed_observation_ids)
    assert ev.evaluated == 1 and ev.by_horizon == {7: 1}
    e = db.execute(select(ForecastEvaluation)).scalar_one()
    assert e.actual_source == "official_usnic" and e.model_version == "v1" and e.actual_observation_date == date(2026, 9, 17)
    assert e.error_km == pytest.approx(float(haversine_km(points[6].predicted_latitude, points[6].predicted_longitude, -57.50, -47.00)))
    assert EvaluationService(db).evaluate().evaluated == 0  # never double-counted

    ForecastService(db, settings).run()
    sets = db.execute(select(ForecastSet).order_by(ForecastSet.id)).scalars().all()
    assert len(sets) == 2 and sets[1].diagnostics["official_entries"] == 2  # old forecast kept, new one appended
    assert db.scalar(select(func.count()).select_from(Forecast)) == 14


def test_registry_transitions(pipeline, settings) -> None:  # type: ignore[no-untyped-def]
    reg = ModelRegistry(pipeline, settings)
    with pytest.raises(RegistryError):
        reg.transition(reg.get("v1"), "candidate", "not allowed")
    assert pipeline.scalar(select(func.count()).select_from(ModelVersion).where(ModelVersion.status == "deployed")) == 1


def test_retraining_skips_when_policy_not_met(pipeline, settings) -> None:  # type: ignore[no-untyped-def]
    run = RetrainingService(pipeline, settings).run()
    assert run.status == "skipped" and "new_evaluations" in run.failure_reason
    assert ModelRegistry(pipeline, settings).get_deployed().version == "v1"


def test_forced_retraining_creates_immutable_challenger(pipeline, settings) -> None:  # type: ignore[no-untyped-def]
    db = pipeline
    add_historical_track(db, "H1", date(2017, 1, 1), 2600, lat0=-64.0, lon0=30.0)
    IngestionService(db, settings).run(document=document(csv_rows(A81_0910)))
    IngestionService(db, settings).run(document=document(csv_rows(A81_0917), datetime(2026, 9, 18, tzinfo=UTC)))
    s = settings.model_copy(update={"retrain_epochs": 1, "retrain_historical_replay_samples": 200})
    run = RetrainingService(db, s).run(trigger="test", force=True)
    assert run.status in ("promoted", "rejected"), run.failure_reason
    reg = ModelRegistry(db, s)
    v2 = reg.get("v2")
    assert v2.parent_version == "v1" and v2.retraining_run_id == run.id
    assert v2.status == ("deployed" if run.status == "promoted" else "rejected")
    assert reg.get("v1").status == ("archived" if run.status == "promoted" else "deployed")
    assert (Path(s.models_dir) / "v2" / "dataset_manifest.json").exists()
    assert run.decision["protocol"].startswith("historical_test")  # 1 operational label < min samples
    assert set(run.metrics["protocols"]) >= {f"historical_test_{s.historical_validation_end}"}
    assert db.scalar(select(func.count()).select_from(ModelVersion).where(ModelVersion.status == "deployed")) == 1
