"""Copernicus Marine Service providers (wind, surface current, sea ice).

Uses the official ``copernicusmarine`` toolbox (v2) ``open_dataset``, which
reads the ARCO (Zarr) service lazily, so only the requested bbox × period is
transferred — never the global product.

Credentials: ``COPERNICUS_MARINE_USERNAME`` / ``COPERNICUS_MARINE_PASSWORD``
(or the toolbox's own ``COPERNICUSMARINE_SERVICE_USERNAME`` / ``_PASSWORD``),
passed explicitly; nothing is written to disk.

Dataset details were verified against the live catalogue (toolbox 2.4.1,
2026-09-11); see ``SPECS``.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

from ml.environment.providers.base import EnvironmentalProvider, ProviderUnavailableError
from ml.environment.providers.canonical import canonicalize
from ml.environment.types import BBox, EnvGroup, ProviderRole, ProviderSpec

CMEMS = "Copernicus Marine Service (CMEMS)"
SURFACE = "uppermost model level, 0.494 m (depth requested as 0–1 m)"

SPECS: dict[str, ProviderSpec] = {
    s.name: s
    for s in (
        ProviderSpec(
            name="copernicus_marine_wind_nrt", group=EnvGroup.WIND, role=ProviderRole.OPERATIONAL, authority=CMEMS,
            product_id="WIND_GLO_PHY_L4_NRT_012_004", dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
            native_variables={"eastward_wind": "wind_u", "northward_wind": "wind_v"},
            units={"wind_u": "m s-1", "wind_v": "m s-1"}, spatial_resolution_deg=0.125, temporal_resolution="PT1H",
            aggregation="mean of hourly L4 fields over the UTC day", latency_days=1.0, coverage_start=date(2024, 6, 13),
            notes="Blended scatterometer + model 10 m wind (L4). Observation-based: no forecast period.",
        ),
        ProviderSpec(
            name="copernicus_marine_wind_my", group=EnvGroup.WIND, role=ProviderRole.HISTORICAL, authority=CMEMS,
            product_id="WIND_GLO_PHY_L4_MY_012_006", dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
            native_variables={"eastward_wind": "wind_u", "northward_wind": "wind_v"},
            units={"wind_u": "m s-1", "wind_v": "m s-1"}, spatial_resolution_deg=0.125, temporal_resolution="PT1H",
            aggregation="mean of hourly L4 fields over the UTC day", latency_days=120.0, coverage_start=date(2007, 1, 11),
            notes="Reprocessed record (0.125°, from 2007-01-11). Earlier years exist only at 0.25° in a separate dataset.",
        ),
        ProviderSpec(
            name="copernicus_marine_current_anfc", group=EnvGroup.CURRENT, role=ProviderRole.OPERATIONAL, authority=CMEMS,
            product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024", dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            native_variables={"uo": "current_u", "vo": "current_v"},
            units={"current_u": "m s-1", "current_v": "m s-1"}, spatial_resolution_deg=1 / 12, temporal_resolution="P1D",
            aggregation="daily mean (native)", latency_days=1.0, coverage_start=date(2022, 6, 1), depth=SURFACE,
            supports_forecast=True, forecast_lead_days=9,
            notes="Global Ocean Physics Analysis and Forecast; grid limited to >= 80°S.",
        ),
        ProviderSpec(
            name="copernicus_marine_current_my", group=EnvGroup.CURRENT, role=ProviderRole.HISTORICAL, authority=CMEMS,
            product_id="GLOBAL_MULTIYEAR_PHY_001_030", dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            native_variables={"uo": "current_u", "vo": "current_v"},
            units={"current_u": "m s-1", "current_v": "m s-1"}, spatial_resolution_deg=1 / 12, temporal_resolution="P1D",
            aggregation="daily mean (native)", latency_days=80.0, coverage_start=date(1993, 1, 1), depth=SURFACE,
            notes="GLORYS12 reanalysis; grid limited to >= 80°S.",
        ),
        ProviderSpec(
            name="copernicus_marine_sea_ice_anfc", group=EnvGroup.SEA_ICE, role=ProviderRole.OPERATIONAL, authority=CMEMS,
            product_id="GLOBAL_ANALYSISFORECAST_PHY_001_024", dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            native_variables={"siconc": "sea_ice_concentration"}, units={"sea_ice_concentration": "1"},
            spatial_resolution_deg=1 / 12, temporal_resolution="P1D", aggregation="daily mean (native)", latency_days=1.0,
            coverage_start=date(2022, 6, 1), supports_forecast=True, forecast_lead_days=9,
            notes="sea_ice_area_fraction from the global model; covers the Southern Ocean to 80°S.",
        ),
        ProviderSpec(
            name="copernicus_marine_sea_ice_my", group=EnvGroup.SEA_ICE, role=ProviderRole.HISTORICAL, authority=CMEMS,
            product_id="GLOBAL_MULTIYEAR_PHY_001_030", dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            native_variables={"siconc": "sea_ice_concentration"}, units={"sea_ice_concentration": "1"},
            spatial_resolution_deg=1 / 12, temporal_resolution="P1D", aggregation="daily mean (native)", latency_days=80.0,
            coverage_start=date(1993, 1, 1),
        ),
    )
}


def _credentials(username: str | None, password: str | None) -> tuple[str | None, str | None]:
    user = username or os.environ.get("COPERNICUS_MARINE_USERNAME") or os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd = password or os.environ.get("COPERNICUS_MARINE_PASSWORD") or os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    return user, pwd


class CopernicusMarineProvider(EnvironmentalProvider):
    def __init__(self, spec: ProviderSpec, username: str | None = None, password: str | None = None, opener: Any = None) -> None:
        self.spec = spec
        self._username, self._password = _credentials(username, password)
        self._opener = opener  # injectable for tests

    def is_configured(self) -> tuple[bool, str]:
        if self._opener is not None:
            return True, "ok"
        try:
            import copernicusmarine  # noqa: F401
        except ImportError:
            return False, "copernicusmarine package not installed"
        if not (self._username and self._password):
            return False, "COPERNICUS_MARINE_USERNAME / COPERNICUS_MARINE_PASSWORD not set"
        return True, "ok"

    def _open(self, bbox: BBox, start: date, end: date) -> Any:
        ok, reason = self.is_configured()
        if not ok:
            raise ProviderUnavailableError(reason)
        if self._opener is not None:
            opener = self._opener
        else:
            import copernicusmarine

            opener = copernicusmarine.open_dataset
        kwargs: dict[str, Any] = dict(
            dataset_id=self.spec.dataset_id,
            variables=list(self.spec.native_variables),
            minimum_longitude=bbox.lon_min,
            maximum_longitude=bbox.lon_max,
            minimum_latitude=max(bbox.lat_min, -80.0) if self.spec.group != EnvGroup.WIND else bbox.lat_min,
            maximum_latitude=bbox.lat_max,
            start_datetime=f"{start.isoformat()}T00:00:00",
            end_datetime=f"{end.isoformat()}T23:59:59",
            username=self._username,
            password=self._password,
        )
        if self.spec.depth:
            kwargs.update(minimum_depth=0.0, maximum_depth=1.0)
        try:
            return opener(**kwargs)
        except ProviderUnavailableError:
            raise
        except Exception as exc:  # network / auth / service errors from the toolbox
            raise ProviderUnavailableError(f"{self.spec.dataset_id}: {type(exc).__name__}: {exc}") from exc

    def fetch_region(self, bbox: BBox, start: date, end: date) -> Any:
        raw = self._open(bbox, start, end)
        return canonicalize(
            raw,
            self.spec.native_variables,
            hourly=self.spec.temporal_resolution == "PT1H",
            attrs={"provider": self.spec.name, "dataset_id": self.spec.dataset_id, "depth": self.spec.depth or "surface/10 m"},
        )

    def fetch_forecast(self, bbox: BBox, issued_on: date, lead_days: int) -> Any:
        """Forecast days issued_on+1 .. issued_on+lead_days, as served *now*.

        Only meaningful when called on the issue date itself: the service does
        not keep past forecast runs, so this can only archive what is available
        at forecast time (never used to reconstruct historical forecasts).
        """
        if not self.spec.supports_forecast:
            raise ProviderUnavailableError(f"{self.spec.name} has no forecast period")
        lead = min(lead_days, self.spec.forecast_lead_days)
        return self.fetch_region(bbox, issued_on + timedelta(days=1), issued_on + timedelta(days=lead))
