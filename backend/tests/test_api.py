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
from tests.helpers import add_historical_track, document, write_tiny_base

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
    assert {m["horizonDays"] for m in detail["metrics"]} == {1, 3, 7}
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
