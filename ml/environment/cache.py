"""Tile x month NetCDF cache for environmental fields.

Gridded products are global and large; a forecast needs a handful of points.
The cache therefore fetches only a **5°×5° tile (plus a two-cell halo for
interpolation) for one calendar month**, stores it as NetCDF with a JSON
sidecar (source, dataset, bbox, period, sha256, last valid day), and serves
every later point lookup in that tile/month from disk.

A month that was fetched before all its days were published is marked
``complete = false`` and is refreshed — at most once per ``refresh_after_hours`` —
when a later day is requested. Files are written atomically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from calendar import monthrange
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ml.environment.providers.base import EnvironmentalProvider
from ml.environment.types import BBox

TILE_DEG = 5.0


@dataclass(frozen=True)
class CacheRecord:
    provider: str
    dataset_id: str
    variables: list[str]
    tile_key: str
    bbox: dict[str, float]
    period_start: str
    period_end: str
    path: str
    sha256: str
    bytes: int
    fetched_at: str
    max_valid_date: str | None
    complete: bool
    spatial_resolution_deg: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tile_origin(lat: float, lon: float, size: float = TILE_DEG) -> tuple[float, float]:
    lat0 = math.floor(lat / size) * size
    lon0 = math.floor(lon / size) * size
    return max(-90.0, min(lat0, 90.0 - size)), max(-180.0, min(lon0, 180.0 - size))


def tile_key(lat0: float, lon0: float) -> str:
    ns = "S" if lat0 < 0 else "N"
    ew = "W" if lon0 < 0 else "E"
    return f"{ns}{abs(int(lat0)):02d}{ew}{abs(int(lon0)):03d}"


def tile_bbox(lat0: float, lon0: float, halo: float, size: float = TILE_DEG) -> BBox:
    return BBox(
        lat_min=max(-90.0, lat0 - halo),
        lat_max=min(90.0, lat0 + size + halo),
        lon_min=max(-180.0, lon0 - halo),
        lon_max=min(180.0, lon0 + size + halo),
    )


def month_period(day: date) -> tuple[date, date]:
    return date(day.year, day.month, 1), date(day.year, day.month, monthrange(day.year, day.month)[1])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EnvironmentalCache:
    def __init__(
        self,
        root: Path,
        on_fetched: Callable[[CacheRecord], None] | None = None,
        max_open: int = 64,
        refresh_after_hours: float = 6.0,
        today: Callable[[], date] | None = None,
    ) -> None:
        self.root = Path(root)
        self.on_fetched = on_fetched
        self.max_open = max_open
        self.refresh_after = timedelta(hours=refresh_after_hours)
        self._today = today or (lambda: datetime.now(UTC).date())
        self._open: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.fetch_count = 0

    # --------------------------------------------------------------- locations
    def paths(self, provider: EnvironmentalProvider, lat: float, lon: float, day: date) -> tuple[Path, Path, str, BBox, date, date]:
        lat0, lon0 = tile_origin(lat, lon)
        key = tile_key(lat0, lon0)
        halo = 2 * provider.spec.spatial_resolution_deg
        start, end = month_period(day)
        base = self.root / provider.spec.name / provider.spec.dataset_id.replace("/", "_") / key
        nc = base / f"{start:%Y-%m}.nc"
        return nc, nc.with_suffix(".json"), key, tile_bbox(lat0, lon0, halo), start, end

    # ------------------------------------------------------------------- access
    def get(self, provider: EnvironmentalProvider, lat: float, lon: float, day: date):  # type: ignore[no-untyped-def]
        """Dataset for the tile/month containing (lat, lon, day), fetching it if needed."""
        import xarray as xr

        nc, meta_path, key, bbox, start, end = self.paths(provider, lat, lon, day)
        cache_id = str(nc)
        with self._lock:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
            if nc.exists() and meta and self._fresh_enough(meta, day):
                if cache_id not in self._open:
                    self._open[cache_id] = xr.load_dataset(nc)
                    while len(self._open) > self.max_open:
                        self._open.popitem(last=False)
                self._open.move_to_end(cache_id)
                return self._open[cache_id], cache_id
            ds = self._fetch_and_store(provider, nc, meta_path, key, bbox, start, end)
            self._open[cache_id] = ds
            return ds, cache_id

    def _fresh_enough(self, meta: dict[str, Any], day: date) -> bool:
        if meta.get("complete"):
            return True
        max_valid = meta.get("max_valid_date")
        if max_valid and day <= date.fromisoformat(max_valid):
            return True
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        return datetime.now(UTC) - fetched_at < self.refresh_after

    def _fetch_and_store(self, provider, nc: Path, meta_path: Path, key: str, bbox: BBox, start: date, end: date):  # type: ignore[no-untyped-def]
        import numpy as np

        fetch_end = min(end, self._today())
        ds = provider.fetch_region(bbox, start, fetch_end)
        self.fetch_count += 1
        times = ds["time"].values.astype("datetime64[D]")
        valid_days = [
            t for i, t in enumerate(times)
            if any(np.isfinite(ds[v].values[i]).any() for v in provider.spec.native_variables.values())
        ]
        max_valid = str(max(valid_days)) if valid_days else None
        complete = max_valid is not None and date.fromisoformat(max_valid) >= end

        nc.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=nc.parent, suffix=".part")
        os.close(fd)
        ds.to_netcdf(tmp)
        os.replace(tmp, nc)
        record = CacheRecord(
            provider=provider.spec.name,
            dataset_id=provider.spec.dataset_id,
            variables=list(provider.spec.native_variables),
            tile_key=key,
            bbox=bbox.as_dict(),
            period_start=start.isoformat(),
            period_end=end.isoformat(),
            path=str(nc),
            sha256=_sha256(nc),
            bytes=nc.stat().st_size,
            fetched_at=datetime.now(UTC).isoformat(),
            max_valid_date=max_valid,
            complete=complete,
            spatial_resolution_deg=provider.spec.spatial_resolution_deg,
        )
        meta_path.write_text(json.dumps(record.as_dict(), indent=2))
        if self.on_fetched:
            self.on_fetched(record)
        return ds

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_text()) for p in self.root.rglob("*.json")]
