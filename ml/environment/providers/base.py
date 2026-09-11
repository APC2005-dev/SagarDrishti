"""Provider interface: every source returns the same canonical daily grid.

``fetch_region`` must return an ``xarray.Dataset`` with

* dims ``(time, lat, lon)``; ``time`` = UTC midnight of each day (daily values),
  ``lat`` ascending, ``lon`` ascending in [-180, 180]
* canonical variable names (``wind_u``, ``current_v``, ``sea_ice_concentration`` …)
  in the units listed in ``spec.units``
* NaN where the source has no value (land, ice-shelf, outside coverage) — never a fill value

so the sampler, cache and feature builder never contain source-specific code.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import TYPE_CHECKING

from ml.environment.types import BBox, ProviderSpec

if TYPE_CHECKING:
    import xarray as xr


class ProviderUnavailableError(RuntimeError):
    """Credentials missing, client library missing, or the remote service failed."""


class EnvironmentalProvider(ABC):
    spec: ProviderSpec

    def last_available_date(self, as_of: date) -> date:
        """Latest field date that would have been published by ``as_of`` (latency rule)."""
        return as_of - timedelta(days=math.ceil(self.spec.latency_days))

    def covers(self, day: date) -> bool:
        return day >= self.spec.coverage_start and (self.spec.coverage_end is None or day <= self.spec.coverage_end)

    def is_configured(self) -> tuple[bool, str]:
        """(ready, reason). Overridden by providers that need credentials."""
        return True, "ok"

    @abstractmethod
    def fetch_region(self, bbox: BBox, start: date, end: date) -> xr.Dataset:
        """Canonical daily fields for ``bbox`` over ``[start, end]`` (inclusive)."""

    def fetch_forecast(self, bbox: BBox, issued_on: date, lead_days: int) -> xr.Dataset:
        """Forecast fields issued on ``issued_on`` for lead days 1..lead_days (if supported)."""
        raise ProviderUnavailableError(f"{self.spec.name} does not provide forecasts")
