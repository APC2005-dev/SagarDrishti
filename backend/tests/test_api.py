"""HTTP API against a seeded PostGIS database."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.session import get_async_session
from app.main import app
from app.services.forecast_service import ForecastService
from app.services.ingestion_service import IngestionService
from app.services.model_registry import ModelRegistry
from tests.helpers import add_historical_track, base_p90, document, write_tiny_base

pytestmark = [pytest.mark.db, pytest.mark.tf]


@pytest.fixture
def seeded(db, settings, fixture_csv):  # type: ignore[no-untyped-def]
    write_tiny_base(Path(settings.models_dir))
    add_historical_track(db, "A81", date(2026, 8, 1), 30)
    ModelRegistry(db, settings).ensure_v1_bootstrap()
    db.commit()
    IngestionService(db, settings).run(document=document(fixture_csv))
    ForecastService(db, settings).run()
    return db


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


async def test_health(client, engine) -> None:  # type: ignore[no-untyped-def]
    r = await client.get("/api/v1/health/live")
    assert r.status_code == 200 and r.headers["x-request-id"]
    r = await client.get("/api/v1/health/ready")
    assert r.json()["checks"]["database"]["ok"] is True


async def test_icebergs(client, seeded) -> None:  # type: ignore[no-untyped-def]
    r = await client.get("/api/v1/icebergs", params={"limit": 5})
    body = r.json()
    assert r.status_code == 200 and body["total"] == 33 and len(body["items"]) == 5
    assert body["items"][0]["icebergId"] == "D15A"  # ordered by area
    first = body["items"][0]
    assert first["provenance"] == "official_usnic" and first["lastUpdate"] == "2026-09-10"
    r = await client.get("/api/v1/icebergs", params={"q": "a8"})
    assert {i["icebergId"] for i in r.json()["items"]} == {"A81", "A83", "A84", "A85"}


async def test_iceberg_detail_position_history(client, seeded) -> None:  # type: ignore[no-untyped-def]
    d = (await client.get("/api/v1/icebergs/a-81")).json()
    assert d["icebergId"] == "A81" and d["officialObservationCount"] == 1 and d["historicalObservationCount"] == 30
    assert d["hasForecast"] and d["latestForecastModelVersion"] == "v1"
    p = (await client.get("/api/v1/icebergs/A81/position")).json()
    assert (p["latitude"], p["longitude"], p["provenance"]) == (-57.36, -47.22, "official_usnic")
    h = (await client.get("/api/v1/icebergs/A81/history")).json()
    assert h["total"] == 1
    h = (await client.get("/api/v1/icebergs/A81/history", params={"include_historical": True})).json()
    assert h["total"] == 31
    assert (await client.get("/api/v1/icebergs/ZZ99")).status_code == 404
    assert (await client.get("/api/v1/icebergs/ZZ99/position")).status_code == 404


@pytest.mark.parametrize("horizon", [1, 3, 7])
async def test_forecast_horizon_filter(client, seeded, horizon) -> None:  # type: ignore[no-untyped-def]
    r = await client.get("/api/v1/icebergs/A81/forecast", params={"horizon": horizon})
    body = r.json()
    assert r.status_code == 200
    assert [p["horizonDays"] for p in body["points"]] == list(range(1, horizon + 1))
    assert body["modelVersion"] == "v1" and body["isChampion"] and body["anchor"]["provenance"] == "official_usnic"
    assert len(body["inputEntries"]) == 14
    assert all(p["provenance"] == "predicted" for p in body["points"])


async def test_forecast_errors(client, seeded) -> None:  # type: ignore[no-untyped-def]
    assert (await client.get("/api/v1/icebergs/A81/forecast", params={"horizon": 8})).status_code == 422
    assert (await client.get("/api/v1/icebergs/D37/forecast")).status_code == 404  # no history -> no forecast


async def test_forecast_listing(client, seeded) -> None:  # type: ignore[no-untyped-def]
    latest = (await client.get("/api/v1/forecasts/latest", params={"horizon": 3})).json()
    assert len(latest) == 1 and len(latest[0]["points"]) == 3 and latest[0]["inputEntries"] is None
    rows = (await client.get("/api/v1/forecasts", params={"iceberg": "A81", "horizon": 7})).json()
    assert rows["total"] == 1 and rows["items"][0]["forecastDate"] == "2026-09-17"
    rows = (await client.get("/api/v1/forecasts", params={"model_version": "v1", "max_horizon": 3})).json()
    assert rows["total"] == 3


async def test_models(client, seeded) -> None:  # type: ignore[no-untyped-def]
    versions = (await client.get("/api/v1/models")).json()
    assert [v["version"] for v in versions] == ["base", "v1"]
    cur = (await client.get("/api/v1/models/current")).json()
    assert cur["version"] == "v1" and cur["architectureVersion"] == "gru_entry14_to_day7_v1"
    detail = (await client.get("/api/v1/models/v1")).json()
    assert [e["toStatus"] for e in detail["statusHistory"]] == ["candidate", "validated", "deployed"]
    assert {m["horizonDays"] for m in detail["metrics"]} == set(base_p90())
    assert detail["modelType"] == "trajectory" and detail["featureSchemaVersion"] == "trajectory_v1"
    assert detail["featureSchema"]["features"][:2] == ["relative_x_km", "relative_y_km"]
    assert versions[0]["modelType"] == "base"
    lineage = (await client.get("/api/v1/models/lineage")).json()
    assert {n["version"]: n["parentVersion"] for n in lineage} == {"base": None, "v1": "base"}
    ev = (await client.get("/api/v1/models/v1/evaluations")).json()
    assert ev["total"] == 0 and ev["byHorizon"] == []
    assert (await client.get("/api/v1/models/v99")).status_code == 404


async def test_operations_feeds_overview(client, seeded) -> None:  # type: ignore[no-untyped-def]
    ing = (await client.get("/api/v1/operations/ingestion")).json()
    assert ing["latestRun"]["newObservations"] == 33 and ing["latestOfficialObservationDate"] == "2026-09-10"
    assert ing["feedState"] in ("LIVE", "DEGRADED")
    feeds = (await client.get("/api/v1/feeds")).json()
    assert feeds[0]["id"] == "usnic_antarctic_icebergs" and feeds[0]["recordCount"] == 33 and feeds[0]["checksumSha256"]
    fr = (await client.get("/api/v1/operations/forecasting")).json()
    assert fr[0]["forecastSetsCreated"] == 1
    rt = (await client.get("/api/v1/operations/retraining")).json()
    assert rt["eligibility"]["eligible"] is False and rt["policy"]["promotion"]["primary_horizon"] == 7
    ov = (await client.get("/api/v1/overview")).json()
    assert ov["activeIcebergs"] == 33 and ov["champion"]["version"] == "v1" and ov["iceberg" + "sWithActiveForecasts"] == 1


async def test_openapi_documents_error_responses(client) -> None:  # type: ignore[no-untyped-def]
    spec = (await client.get("/openapi.json")).json()
    op = spec["paths"]["/api/v1/icebergs/{iceberg_id}/forecast"]["get"]
    assert "404" in op["responses"] and "422" in op["responses"]
    assert any(p["name"] == "horizon" for p in op["parameters"])


# ------------------------------------------------- two core families at once
@pytest.fixture
def both_families(seeded, settings):  # type: ignore[no-untyped-def]
    """Trajectory AND sea-ice each deployed — the state that used to 500.

    ``deployed`` is unique per family, not globally, so this is a valid state:
    it must not raise MultipleResultsFound anywhere.
    """
    from app.services.seaice_registry import SeaIceRegistry
    from tests.helpers import write_tiny_seaice_base

    write_tiny_seaice_base(Path(settings.models_dir))
    registry = SeaIceRegistry(seeded, settings)
    registry.register_base()
    registry.ensure_champion()
    seeded.commit()
    return seeded


async def test_endpoints_survive_two_deployed_families(both_families, client) -> None:  # type: ignore[no-untyped-def]
    """TEST 1 / 11 / 12: no MultipleResultsFound once a second family is deployed."""
    for path in ("/api/v1/health/ready", "/api/v1/overview", "/api/v1/forecasts/latest?horizon=7",
                 "/api/v1/models/current", "/api/v1/models", "/api/v1/models/lineage", "/api/v1/feeds"):
        response = await client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}: {response.text[:300]}"


async def test_health_reports_each_family_separately(both_families, client) -> None:  # type: ignore[no-untyped-def]
    checks = (await client.get("/api/v1/health/ready")).json()["checks"]
    assert checks["model:trajectory"]["version"] == "v1"
    assert checks["model:sea_ice"]["version"] == "sea_ice/base"
    assert checks["model:trajectory"]["ok"] and checks["model:sea_ice"]["ok"]


async def test_champion_is_resolved_per_family(both_families, client) -> None:  # type: ignore[no-untyped-def]
    """TEST 3 / 4: each family resolves to its OWN model, never the other's."""
    trajectory = (await client.get("/api/v1/models/current?family=trajectory")).json()
    seaice = (await client.get("/api/v1/models/current?family=sea_ice")).json()
    assert trajectory["version"] == "v1" and trajectory["architecture"] == "GRU"
    assert seaice["version"] == "sea_ice/base" and seaice["architecture"] == "UNetResidual"
    # The default (no family) is the trajectory product, unchanged for existing clients.
    assert (await client.get("/api/v1/models/current")).json()["version"] == "v1"

    champions = (await client.get("/api/v1/models/champions")).json()
    assert set(champions) == {"trajectory", "sea_ice"}
    assert champions["sea_ice"]["version"] == "sea_ice/base"


async def test_model_listings_do_not_mix_families(both_families, client) -> None:  # type: ignore[no-untyped-def]
    trajectory = (await client.get("/api/v1/models?family=trajectory")).json()
    seaice = (await client.get("/api/v1/models?family=sea_ice")).json()
    assert {m["version"] for m in trajectory} == {"base", "v1"}
    assert all(m["version"].startswith("sea_ice/") for m in seaice)


async def test_seaice_endpoints(both_families, client) -> None:  # type: ignore[no-untyped-def]
    """TEST 9 / 10: the sea-ice API resolves the sea-ice champion, not the GRU."""
    status = (await client.get("/api/v1/sea-ice/status")).json()
    assert status["model"]["version"] == "sea_ice/base"
    assert status["model"]["architecture"] == "UNetResidual"
    assert status["model"]["inputWindowEntries"] == 7  # entries, not calendar days
    assert status["datasetId"] == "osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m"
    # No sea-ice data in this fixture: the reason is explicit, no fake values.
    assert status["observationCount"] == 0
    assert status["windowComplete"] is False
    assert "7 chronological entries" in status["forecastUnavailableReason"]
    assert status["latestForecasts"] == []
    assert (await client.get("/api/v1/sea-ice/latest")).status_code == 404
    assert (await client.get("/api/v1/sea-ice/forecast?horizon=7")).status_code == 404
    assert (await client.get("/api/v1/sea-ice/model")).json()["version"] == "sea_ice/base"


async def test_feeds_lists_seaice_as_a_core_feed(both_families, client) -> None:  # type: ignore[no-untyped-def]
    """TEST 6 / 24: the sea-ice source is a CORE feed with the notebook's dataset."""
    feeds = {f["id"]: f for f in (await client.get("/api/v1/feeds")).json()}
    seaice = feeds["copernicus_seaice_osisaf"]
    assert seaice["category"] == "sea_ice"  # core, not "environmental"
    assert seaice["sourceUrl"] == "osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m"
    assert feeds["usnic_antarctic_icebergs"]["category"] == "iceberg"
    # The environmental sea-ice FEATURE provider is a separate, distinct entry.
    assert feeds["env_operational_sea_ice"]["category"] == "environmental"
