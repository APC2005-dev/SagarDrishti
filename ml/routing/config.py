"""Route-planner configuration — the notebook's ``DemoConfig``, field for field.

Every default below is the value the supplied route-planning notebook used. They
are **illustrative software-test assumptions, not validated vessel ratings or
calibrated safety distances**; that wording is the notebook's own and is kept
because the numbers carry exactly that status in production too.

Changing any of these materially (objective, weights, buffers, speeds, grid
behaviour) changes the routes produced and therefore requires a new
``ROUTE_PLANNER_VERSION``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Bump when the objective, constraints, grid behaviour, environmental mapping or
# curvature definition materially change. This is algorithmic versioning: the
# route engine is a time-dependent A*, not a trained model.
ROUTE_PLANNER_VERSION = "route_planner_v1"

# The notebook's regional routing domain (Weddell Sea development scenario):
# west, east, south, north. Ports outside this box cannot be routed.
DOMAIN_WEST, DOMAIN_EAST = -65.0, -30.0
DOMAIN_SOUTH, DOMAIN_NORTH = -75.0, -55.0

# Forecast grid spacing the notebook asserts and the sea-ice model produces.
GRID_RESOLUTION_DEG = 0.5

# Data-coverage rule, not a vessel safety threshold: at least this fraction of
# valid original pixels on EACH input day.
MIN_HISTORY_VALID_FRACTION = 0.95

KNOTS_TO_M_PER_S = 0.514444444444


@dataclass(frozen=True)
class RouteConfig:
    max_through_water_knots: float = 10.0
    ground_action_knots: tuple[float, ...] = (4.0, 6.0, 8.0)
    max_sic: float = 0.30
    ice_speed_reduction: float = 0.50
    forecast_buffer_km: float = 50.0
    unresolved_buffer_km: float = 100.0
    hotel_proxy_per_hour: float = 0.20
    time_weight: float = 0.50
    fuel_weight: float = 0.35
    ice_weight: float = 0.15
    time_reference_hours: float = 24.0
    fuel_reference: float = 24.0
    ice_reference_hours: float = 24.0
    horizon_hours: int = 168
    max_expansions: int = 60_000
    max_runtime_seconds: float = 180.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
