"""Basemap tile proxy: NASA GIBS Blue Marble in native EPSG:3031.

Tiles are fetched once from GIBS, validated, and cached on disk; afterwards they
are served locally (the map keeps working if GIBS is unreachable). The imagery
is a static cloud-free composite used purely as a visual backdrop — it is not
an observation and nothing in the ML pipeline reads it.

Tile geometry of the GIBS EPSG:3031 "500m" matrix set (verified against the live
service): extent ±4 194 304 m, 512 px tiles, level z has 2^(z+1) × 2^(z+1)
tiles at 8192 / 2^z m per pixel, row 0 at +y (towards 0° longitude), z = 0…4.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import Response

from app.api.deps import SettingsDep
from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.common import ERROR_RESPONSES, ApiModel

router = APIRouter(prefix="/basemap", tags=["basemap"])
log = get_logger(__name__)

EXTENT_M = 4_194_304.0
TILE_PX = 512
ATTRIBUTION = "Imagery: NASA Blue Marble via NASA GIBS (EPSG:3031)"
LAYERS: dict[str, dict[str, str | int]] = {
    "BlueMarble_ShadedRelief_Bathymetry": {"tile_matrix_set": "500m", "max_zoom": 4, "title": "Blue Marble shaded relief + bathymetry"},
    "BlueMarble_NextGeneration": {"tile_matrix_set": "500m", "max_zoom": 4, "title": "Blue Marble Next Generation"},
}
DEFAULT_LAYER = "BlueMarble_ShadedRelief_Bathymetry"
CACHE_HEADERS = {"Cache-Control": "public, max-age=2592000, immutable"}


class BasemapLayer(ApiModel):
    id: str
    title: str
    max_zoom: int


class BasemapInfo(ApiModel):
    enabled: bool
    crs: str
    extent_m: float
    tile_size: int
    matrix_size_formula: str
    default_layer: str
    layers: list[BasemapLayer]
    attribution: str


def matrix_size(z: int) -> int:
    return 2 ** (z + 1)


def cache_path(settings: Settings, layer: str, z: int, row: int, col: int) -> Path:
    return Path(settings.basemap_cache_dir) / layer / str(z) / str(row) / f"{col}.jpeg"


async def _fetch_upstream(url: str, timeout: float) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
    if resp.status_code != 200:
        raise HTTPException(502, f"basemap upstream returned HTTP {resp.status_code}")
    return resp.content


@router.get("", response_model=BasemapInfo, summary="Basemap tile geometry and attribution")
async def info(settings: SettingsDep) -> BasemapInfo:
    return BasemapInfo(
        enabled=settings.basemap_enabled,
        crs="EPSG:3031",
        extent_m=EXTENT_M,
        tile_size=TILE_PX,
        matrix_size_formula="2^(z+1)",
        default_layer=DEFAULT_LAYER,
        layers=[BasemapLayer(id=k, title=str(v["title"]), max_zoom=int(v["max_zoom"])) for k, v in LAYERS.items()],
        attribution=ATTRIBUTION,
    )


@router.get(
    "/{layer}/{z}/{row}/{col}.jpeg",
    responses={**ERROR_RESPONSES, 200: {"content": {"image/jpeg": {}}}, 502: {"description": "Upstream imagery unavailable"}},
    response_class=Response,
    summary="One 512 px EPSG:3031 basemap tile (cached)",
)
async def tile(
    layer: str,
    z: Annotated[int, PathParam(ge=0, le=4)],
    row: Annotated[int, PathParam(ge=0)],
    col: Annotated[int, PathParam(ge=0)],
    settings: SettingsDep,
) -> Response:
    if not settings.basemap_enabled:
        raise HTTPException(404, "basemap disabled")
    spec = LAYERS.get(layer)
    if spec is None:
        raise HTTPException(404, f"unknown basemap layer {layer!r}")
    if z > int(spec["max_zoom"]) or row >= matrix_size(z) or col >= matrix_size(z):
        raise HTTPException(422, f"tile {z}/{row}/{col} outside the {matrix_size(z)}x{matrix_size(z)} matrix")

    path = cache_path(settings, layer, z, row, col)
    if path.exists():
        return Response(path.read_bytes(), media_type="image/jpeg", headers={**CACHE_HEADERS, "X-Basemap-Cache": "hit"})

    url = settings.basemap_url_template.format(layer=layer, tile_matrix_set=spec["tile_matrix_set"], z=z, row=row, col=col)
    try:
        content = await _fetch_upstream(url, settings.basemap_timeout_seconds)
    except httpx.HTTPError as exc:
        log.warning("basemap_upstream_unreachable", url=url, error=str(exc))
        raise HTTPException(502, "basemap upstream unreachable") from exc
    if not content.startswith(b"\xff\xd8"):
        raise HTTPException(502, "basemap upstream returned a non-JPEG payload")

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    with os.fdopen(fd, "wb") as fh:
        fh.write(content)
    os.replace(tmp, path)
    return Response(content, media_type="image/jpeg", headers={**CACHE_HEADERS, "X-Basemap-Cache": "miss"})
