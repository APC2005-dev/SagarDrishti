"""Route curvature — ported verbatim from the notebook's ``curvature_metrics``.

Curvature is **computed from the A* route geometry**, not predicted by any
model. A waypoint polyline is not a smooth ship manoeuvre, so the result is a
discrete proxy:

    signed local heading change / mean adjacent geodesic segment lengths

in rad/km. It is explicitly NOT a physical turning radius, and the route's
source and destination alone do not determine it — the environment the search
routed around does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyproj import Geod

GEOD = Geod(ellps="WGS84")

CURVATURE_DEFINITION = (
    "Signed local heading change / mean adjacent geodesic lengths; "
    "discrete proxy, not physical turning radius."
)
DETOUR_DEFINITION = (
    "Route length / shortest geodesic between grid endpoints. "
    "The geodesic is not a validated feasible route."
)


def curvature_metrics(
    waypoints: list[dict[str, Any]], request: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``waypoints`` are the route's ordered points with latitude/longitude.

    Zero-length waits are excluded from the geometry; no position is moved.
    """
    coords = np.array([[float(w["longitude"]), float(w["latitude"])] for w in waypoints], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite route coordinates.")
    if (np.abs(coords[:, 0]) > 180).any() or (np.abs(coords[:, 1]) > 90).any():
        raise ValueError("Invalid lat/lon.")
    keep = [0]
    for i in range(1, len(waypoints)):
        a, b = coords[keep[-1]], coords[i]
        if GEOD.inv(a[0], a[1], b[0], b[1])[2] > 0.001:
            keep.append(i)
    if len(keep) < 2:
        raise ValueError("No nonzero route movement to analyse.")
    lon = coords[keep, 0]
    lat = coords[keep, 1]
    az, back, metres = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    length = np.asarray(metres) / 1000
    if (length <= 0).any():
        raise ValueError("Degenerate route segment.")
    dist = np.r_[0.0, np.cumsum(length)]

    turns: list[dict[str, Any]] = []
    for j in range(1, len(keep) - 1):
        incoming = (back[j - 1] + 180) % 360  # travel bearing at the joint, not at the prior vertex
        outgoing = az[j] % 360
        turn = (outgoing - incoming + 180) % 360 - 180
        span = (length[j - 1] + length[j]) / 2
        turns.append(
            {
                "original_waypoint_index": int(keep[j]),
                "latitude": float(lat[j]),
                "longitude": float(lon[j]),
                "distance_from_start_km": float(dist[j]),
                "signed_turn_deg": float(turn),
                "absolute_turn_deg": float(abs(turn)),
                "adjacent_mean_length_km": float(span),
                "signed_discrete_curvature_rad_per_km": float(np.deg2rad(turn) / span),
                "absolute_discrete_curvature_rad_per_km": float(abs(np.deg2rad(turn)) / span),
            }
        )

    total = float(length.sum())
    direct = float(GEOD.inv(lon[0], lat[0], lon[-1], lat[-1])[2] / 1000)
    total_turn = float(sum(t["absolute_turn_deg"] for t in turns))
    stats: dict[str, Any] = {
        "route_length_km": total,
        "grid_endpoint_geodesic_km": direct,
        "detour_ratio": total / direct if direct > 1e-9 else None,
        "extra_distance_vs_grid_geodesic_km": total - direct,
        "extra_distance_vs_grid_geodesic_pct": 100 * (total / direct - 1) if direct > 1e-9 else None,
        "total_absolute_turn_deg": total_turn,
        "maximum_turn_deg": max((t["absolute_turn_deg"] for t in turns), default=0.0),
        "maximum_discrete_curvature_rad_per_km": max(
            (t["absolute_discrete_curvature_rad_per_km"] for t in turns), default=0.0
        ),
        "total_turn_radians_per_route_km": float(np.deg2rad(total_turn) / total),
        "removed_stationary_waypoints": int(len(waypoints) - len(keep)),
        "curvature_definition": CURVATURE_DEFINITION,
        "detour_definition": DETOUR_DEFINITION,
        "source_destination_alone_determine_curvature": False,
    }
    if request and "start" in request and "destination" in request:
        a, b = request["start"], request["destination"]
        stats["requested_endpoint_geodesic_km"] = float(GEOD.inv(a[1], a[0], b[1], b[0])[2] / 1000)
        # Set by the route service once the port->grid connectors are actually checked.
        stats["requested_endpoint_connectors_validated"] = bool(
            request.get("connectors_validated", False)
        )
    return stats, turns
