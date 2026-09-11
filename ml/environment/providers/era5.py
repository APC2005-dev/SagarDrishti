"""ERA5 (Copernicus Climate Change Service) historical wind / sea-ice provider.

Request format mirrors the request that was run successfully in the original
research notebook: dataset ``reanalysis-era5-single-levels``, NetCDF,
``download_format = unarchived``, ``area = [North, West, South, East]``.
All 24 hourly steps are requested and averaged over the UTC day.

Credentials: ``CDSAPI_KEY`` (personal access token) and optionally
``CDSAPI_URL`` (default ``https://cds.climate.copernicus.eu/api``). The token is
passed to ``cdsapi.Client`` directly; no ``~/.cdsapirc`` is written.

ERA5 final data lag real time by ~2–3 months (ERA5T by ~5 days), so ERA5 is
configured as a HISTORICAL provider for building training features.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ml.environment.providers.base import EnvironmentalProvider, ProviderUnavailableError
from ml.environment.providers.canonical import canonicalize
from ml.environment.types import BBox, EnvGroup, ProviderRole, ProviderSpec

C3S = "Copernicus Climate Change Service (C3S) — ERA5"
DATASET = "reanalysis-era5-single-levels"
DEFAULT_URL = "https://cds.climate.copernicus.eu/api"

SPECS: dict[str, ProviderSpec] = {
    s.name: s
    for s in (
        ProviderSpec(
            name="era5_wind", group=EnvGroup.WIND, role=ProviderRole.HISTORICAL, authority=C3S, product_id="ERA5",
            dataset_id=DATASET, native_variables={"u10": "wind_u", "v10": "wind_v"},
            units={"wind_u": "m s-1", "wind_v": "m s-1"}, spatial_resolution_deg=0.25, temporal_resolution="PT1H",
            aggregation="mean of 24 hourly reanalysis steps over the UTC day", latency_days=5.0,
            coverage_start=date(1940, 1, 1),
            notes="Request variables 10m_u_component_of_wind / 10m_v_component_of_wind.",
        ),
        ProviderSpec(
            name="era5_sea_ice", group=EnvGroup.SEA_ICE, role=ProviderRole.HISTORICAL, authority=C3S, product_id="ERA5",
            dataset_id=DATASET, native_variables={"siconc": "sea_ice_concentration"}, units={"sea_ice_concentration": "1"},
            spatial_resolution_deg=0.25, temporal_resolution="PT1H",
            aggregation="mean of 24 hourly reanalysis steps over the UTC day", latency_days=5.0,
            coverage_start=date(1940, 1, 1), notes="Request variable sea_ice_cover (siconc, fraction 0-1).",
        ),
    )
}
REQUEST_VARIABLES = {"u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind", "siconc": "sea_ice_cover"}


class Era5Provider(EnvironmentalProvider):
    def __init__(self, spec: ProviderSpec, key: str | None = None, url: str | None = None, client: Any = None) -> None:
        self.spec = spec
        self._key = key or os.environ.get("CDSAPI_KEY")
        self._url = url or os.environ.get("CDSAPI_URL") or DEFAULT_URL
        self._client = client

    def is_configured(self) -> tuple[bool, str]:
        if self._client is not None:
            return True, "ok"
        try:
            import cdsapi  # noqa: F401
        except ImportError:
            return False, "cdsapi package not installed"
        if not self._key:
            return False, "CDSAPI_KEY not set"
        return True, "ok"

    def request_for(self, bbox: BBox, start: date, end: date) -> dict[str, Any]:
        if (start.year, start.month) != (end.year, end.month):
            raise ValueError("ERA5 requests are issued per calendar month")
        days = [(start + timedelta(days=i)).day for i in range((end - start).days + 1)]
        return {
            "product_type": ["reanalysis"],
            "variable": [REQUEST_VARIABLES[v] for v in self.spec.native_variables],
            "year": [f"{start.year}"],
            "month": [f"{start.month:02d}"],
            "day": [f"{d:02d}" for d in days],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [bbox.lat_max, bbox.lon_min, bbox.lat_min, bbox.lon_max],  # North, West, South, East
        }

    def fetch_region(self, bbox: BBox, start: date, end: date) -> Any:
        import xarray as xr

        ok, reason = self.is_configured()
        if not ok:
            raise ProviderUnavailableError(reason)
        client = self._client
        if client is None:
            import cdsapi

            client = cdsapi.Client(url=self._url, key=self._key, quiet=True, progress=False)
        fd, tmp = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        try:
            try:
                client.retrieve(DATASET, self.request_for(bbox, start, end), tmp)
            except Exception as exc:
                raise ProviderUnavailableError(f"ERA5 request failed: {type(exc).__name__}: {exc}") from exc
            with xr.open_dataset(tmp) as src:  # close the handle before the temp file is removed
                raw = src.load()
            return canonicalize(
                raw, self.spec.native_variables, hourly=True,
                attrs={"provider": self.spec.name, "dataset_id": DATASET, "depth": "10 m above surface" if self.spec.group == EnvGroup.WIND else "surface"},
            )
        finally:
            Path(tmp).unlink(missing_ok=True)
