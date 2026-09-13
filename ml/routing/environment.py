"""Build the routing environment the time-dependent A* searches over.

Ported from the notebook's ``prepare_planner`` and its regional-exclusion cell.
Everything the planner consumes is derived from production forecasts:

* sea ice   — the deployed sea-ice champion's D+1/D+3/D+7 grids, anchored on the
              latest official observation, interpolated to hourly;
* icebergs  — the deployed trajectory champion's D+1..D+7 forecasts, swept to
              hourly occupancy buffers;
* currents  — Copernicus Marine surface ``uo``/``vo`` on the same grid;
* static    — Natural Earth land and Antarctic ice shelves, plus data-coverage
              and forecast-validity rules.

Missing data stays missing. Nothing here substitutes zero for an absent field,
and the planner refuses an edge whose cells carry a non-finite value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.interpolate import interp1d

from ml.routing.config import (
    MIN_HISTORY_VALID_FRACTION,
    RouteConfig,
)
from ml.routing.engine import TimeAStar


class RoutingEnvironmentError(RuntimeError):
    """Raised when the environment cannot be built from available data."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StaticExclusions:
    blocked: np.ndarray
    land_blocked: np.ndarray
    shelf_blocked: np.ndarray
    coverage_failed: np.ndarray
    forecast_invalid: np.ndarray

    def counts(self) -> dict[str, int]:
        return {
            "land": int(self.land_blocked.sum()),
            "ice_shelf": int(self.shelf_blocked.sum()),
            "insufficient_coverage": int(self.coverage_failed.sum()),
            "invalid_forecast": int(self.forecast_invalid.sum()),
            "total_blocked": int(self.blocked.sum()),
            "navigable": int((~self.blocked).size - self.blocked.sum()),
        }


def load_regional_polygons(dataset_name: str, region_box: Any) -> Any:
    """Natural Earth 10m physical polygons clipped to the routing region."""
    import shapely
    from cartopy.io import shapereader
    from shapely.ops import unary_union

    path = shapereader.natural_earth(resolution="10m", category="physical", name=dataset_name)
    reader = shapereader.Reader(path)
    try:
        pieces = [g.intersection(region_box) for g in reader.geometries() if g.intersects(region_box)]
    finally:
        reader.close()
    return unary_union(pieces) if pieces else shapely.geometry.GeometryCollection()


def build_static_exclusions(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    history_valid_fraction: np.ndarray,
    forecast_values: np.ndarray,
    resolution_deg: float,
) -> StaticExclusions:
    """Land, ice shelf, data coverage and forecast validity — the notebook's rules.

    A cell is blocked if **any part of it** intersects mapped land or an ice
    shelf: the whole cell area is tested, not just its centre.
    """
    import shapely
    from shapely.geometry import box

    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    half = resolution_deg / 2
    cell_polygons = shapely.box(lon_grid - half, lat_grid - half, lon_grid + half, lat_grid + half)
    region_box = box(
        float(longitudes.min()) - 1,
        float(latitudes.min()) - 1,
        float(longitudes.max()) + 1,
        float(latitudes.max()) + 1,
    )
    land_geometry = load_regional_polygons("land", region_box)
    shelf_geometry = load_regional_polygons("antarctic_ice_shelves_polys", region_box)
    land_blocked = shapely.intersects(cell_polygons, land_geometry)
    shelf_blocked = shapely.intersects(cell_polygons, shelf_geometry)

    # Conservative prototype coverage rule: at least 95% valid original pixels on
    # EACH input day. This is a data-coverage rule, not a vessel safety threshold.
    coverage_failed = ~(np.asarray(history_valid_fraction) >= MIN_HISTORY_VALID_FRACTION)
    forecast_invalid = ~(
        np.isfinite(forecast_values).all(axis=0)
        & (forecast_values >= 0).all(axis=0)
        & (forecast_values <= 1).all(axis=0)
    )
    blocked = land_blocked | shelf_blocked | coverage_failed | forecast_invalid
    return StaticExclusions(blocked, land_blocked, shelf_blocked, coverage_failed, forecast_invalid)


def prepare_planner(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    sic_observed: np.ndarray,
    sic_forecast: np.ndarray,
    sic_forecast_lead_hours: np.ndarray,
    current_times: pd.DatetimeIndex,
    current_u: np.ndarray,
    current_v: np.ndarray,
    iceberg_forecasts: pd.DataFrame,
    unresolved_hazards: pd.DataFrame,
    static_blocked: np.ndarray,
    anchor: Any,
    config: RouteConfig | None = None,
) -> TimeAStar:
    """Assemble hourly environmental fields and iceberg occupancy for the search."""
    c = config or RouteConfig()
    anchor = pd.Timestamp(anchor)
    latitudes = np.asarray(latitudes, float)
    longitudes = np.asarray(longitudes, float)
    times = pd.date_range(anchor, periods=c.horizon_hours + 1, freq="h")

    # --- sea ice: observation at the anchor + the champion's forecast horizons
    sic_times = np.r_[0.0, np.asarray(sic_forecast_lead_hours, float)]
    sic_values = np.concatenate([np.asarray(sic_observed, float)[None], np.asarray(sic_forecast, float)], axis=0)
    if np.any(np.diff(sic_times) <= 0) or sic_times[-1] < c.horizon_hours:
        raise RoutingEnvironmentError(
            "SEA_ICE_FORECAST_UNAVAILABLE",
            "Sea-ice forecast times do not cover the planning horizon",
        )
    sic = interp1d(sic_times, sic_values, axis=0, bounds_error=True)(np.arange(c.horizon_hours + 1))

    # --- ocean currents: must cover the horizon; extrapolation is forbidden
    ct = pd.DatetimeIndex(current_times)
    if len(ct) == 0 or ct.min() > times[0] or ct.max() < times[-1]:
        raise RoutingEnvironmentError(
            "ENVIRONMENT_DATA_UNAVAILABLE",
            "Ocean-current time coverage is insufficient for the planning horizon; "
            "extrapolation is forbidden",
        )
    target = (times - ct[0]).total_seconds().to_numpy() / 3600.0
    source = (ct - ct[0]).total_seconds().to_numpy() / 3600.0
    u = interp1d(source, np.asarray(current_u, float), axis=0, bounds_error=True)(target)
    v = interp1d(source, np.asarray(current_v, float), axis=0, bounds_error=True)(target)

    gx, gy = np.meshgrid(longitudes, latitudes)
    geod = Geod(ellps="WGS84")
    dx, dy = np.diff(longitudes), np.diff(latitudes)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]) and min(dx.min(), dy.min()) > 0):
        raise RoutingEnvironmentError("ROUTING_GRID_UNAVAILABLE", "Expected an ascending regular spatial grid")
    radius = np.zeros(gx.shape)
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        _, _, d = geod.inv(gx, gy, gx + sx * dx[0] / 2, gy + sy * dy[0] / 2)
        radius = np.maximum(radius, d / 1000)

    def distances(lon: float, lat: float) -> np.ndarray:
        return geod.inv(gx, gy, np.full(gx.shape, lon), np.full(gy.shape, lat))[2] / 1000

    # --- iceberg hazards: hourly occupancy swept along the interpolated track
    blocks = np.zeros((c.horizon_hours,) + gx.shape, bool)
    for iceberg_id, track in iceberg_forecasts.groupby("iceberg_id"):
        track = track.sort_values("lead_days")
        if track.lead_days.tolist() != list(range(1, 8)):
            raise RoutingEnvironmentError(
                "TRAJECTORY_FORECAST_UNAVAILABLE", f"Incomplete forecast for {iceberg_id}"
            )
        if not (pd.to_datetime(track.input_end_date) == anchor).all():
            raise RoutingEnvironmentError("ENVIRONMENT_FORECAST_MISMATCH", "Mixed forecast anchors")
        expected = anchor + pd.to_timedelta(track.lead_days, unit="D")
        if not np.array_equal(pd.to_datetime(track.forecast_date).values, expected.values):
            raise RoutingEnvironmentError(
                "ENVIRONMENT_FORECAST_MISMATCH", "Forecast dates and leads disagree"
            )
        xx = np.r_[track.iloc[0].anchor_longitude, track.predicted_longitude.values]
        yy = np.r_[track.iloc[0].anchor_latitude, track.predicted_latitude.values]
        if not (np.isfinite(xx).all() and np.isfinite(yy).all()):
            raise RoutingEnvironmentError("TRAJECTORY_FORECAST_UNAVAILABLE", "Invalid iceberg coordinates")
        hourly = []
        for h in range(c.horizon_hours + 1):
            day = min(h // 24, 6)
            fraction = (h - day * 24) / 24
            bearing, _, d = geod.inv(xx[day], yy[day], xx[day + 1], yy[day + 1])
            lon, lat, _ = geod.fwd(xx[day], yy[day], bearing, d * fraction)
            hourly.append((lon, lat))
        previous = distances(*hourly[0])
        for h in range(c.horizon_hours):
            following = distances(*hourly[h + 1])
            sweep = geod.inv(*hourly[h], *hourly[h + 1])[2] / 2000
            # Covers interpolated motion through each hourly interval plus cell area.
            blocks[h] |= np.minimum(previous, following) <= c.forecast_buffer_km + radius + sweep
            previous = following

    # --- icebergs without enough history to forecast: static exclusion
    unresolved = np.zeros(gx.shape, bool)
    for _, row in unresolved_hazards.iterrows():
        if pd.Timestamp(row["date"]) > anchor:
            raise RoutingEnvironmentError(
                "ENVIRONMENT_FORECAST_MISMATCH", "Future unresolved observation"
            )
        unresolved |= distances(row.longitude, row.latitude) <= c.unresolved_buffer_km + radius

    static = np.asarray(static_blocked, bool) | unresolved
    return TimeAStar(latitudes, longitudes, sic, u, v, static, blocks, anchor, c)
