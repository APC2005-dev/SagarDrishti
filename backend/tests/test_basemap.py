"""Basemap proxy: validation, caching, upstream failure handling (no network, no DB)."""

from __future__ import annotations

import httpx
import pytest

from app.api.v1 import basemap
from app.main import app

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


@pytest.fixture
async def client():  # type: ignore[no-untyped-def]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def upstream(monkeypatch):  # type: ignore[no-untyped-def]
    calls: list[str] = []
    payload = {"body": JPEG}

    async def fake(url: str, timeout: float) -> bytes:
        calls.append(url)
        if isinstance(payload["body"], Exception):
            raise payload["body"]
        return payload["body"]

    monkeypatch.setattr(basemap, "_fetch_upstream", fake)
    return calls, payload


async def test_info(client) -> None:  # type: ignore[no-untyped-def]
    body = (await client.get("/api/v1/basemap")).json()
    assert body["crs"] == "EPSG:3031" and body["extentM"] == 4194304 and body["tileSize"] == 512
    assert body["defaultLayer"] in {layer["id"] for layer in body["layers"]}


async def test_tile_is_fetched_once_then_served_from_cache(client, upstream) -> None:  # type: ignore[no-untyped-def]
    calls, _ = upstream
    url = "/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/2/3/4.jpeg"
    first = await client.get(url)
    assert first.status_code == 200 and first.headers["content-type"] == "image/jpeg"
    assert first.headers["x-basemap-cache"] == "miss" and first.content == JPEG
    assert calls == ["https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/BlueMarble_ShadedRelief_Bathymetry/default/500m/2/3/4.jpeg"]
    second = await client.get(url)
    assert second.headers["x-basemap-cache"] == "hit" and second.content == JPEG
    assert len(calls) == 1


@pytest.mark.parametrize(
    "path,status",
    [
        ("/api/v1/basemap/Not_A_Layer/0/0/0.jpeg", 404),
        ("/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/5/0/0.jpeg", 422),
        ("/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/0/0/2.jpeg", 422),
        ("/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/2/8/0.jpeg", 422),
        ("/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/-1/0/0.jpeg", 422),
    ],
)
async def test_rejects_invalid_tiles(client, upstream, path, status) -> None:  # type: ignore[no-untyped-def]
    assert (await client.get(path)).status_code == status
    assert upstream[0] == []  # never forwarded upstream


async def test_upstream_unreachable(client, upstream) -> None:  # type: ignore[no-untyped-def]
    upstream[1]["body"] = httpx.ConnectError("down")
    r = await client.get("/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/1/0/1.jpeg")
    assert r.status_code == 502


async def test_non_jpeg_upstream_not_cached(client, upstream) -> None:  # type: ignore[no-untyped-def]
    upstream[1]["body"] = b"<html>error</html>"
    url = "/api/v1/basemap/BlueMarble_ShadedRelief_Bathymetry/1/1/1.jpeg"
    assert (await client.get(url)).status_code == 502
    upstream[1]["body"] = JPEG
    assert (await client.get(url)).headers["x-basemap-cache"] == "miss"
