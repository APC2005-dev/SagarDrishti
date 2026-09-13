"""Hourly time-expanded A* — ported verbatim from the route-planning notebook.

The search state is ``(hour, row, col)``, not just position, so edge feasibility
and cost are evaluated against the environment **at the hours the leg actually
occupies**. Waiting in place is a first-class action.

Nothing here is a trained model. The route is produced by graph search; every
metric (distance, duration, fuel proxy, SIC exposure, objective) is accumulated
from the legs the search actually chose.

The mathematics below — edge feasibility, the through-water propulsion check,
the cubic fuel proxy, the ice-speed rule, the weighted objective and the
admissible heuristic — are the notebook's and must not be simplified.

Not for operational navigation: uncalibrated buffers, no bathymetry, no wind or
waves, coarse grid centres and a finite horizon.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod

from ml.routing.config import KNOTS_TO_M_PER_S, RouteConfig


class TimeAStar:
    def __init__(
        self,
        latitude: np.ndarray,
        longitude: np.ndarray,
        sic: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        static: np.ndarray,
        iceberg_intervals: np.ndarray,
        anchor: Any,
        config: RouteConfig | None = None,
    ) -> None:
        self.cfg = config or RouteConfig()
        c = self.cfg
        if c.horizon_hours < 1 or c.max_through_water_knots <= 0:
            raise ValueError("Invalid horizon or vessel speed")
        if not 0 < c.max_sic <= 1 or not 0 <= c.ice_speed_reduction < 1:
            raise ValueError("Invalid illustrative ice configuration")
        if min(c.time_weight, c.fuel_weight, c.ice_weight) < 0 or c.time_weight <= 0:
            raise ValueError("Nonnegative weights and positive time weight required")
        if min(c.time_reference_hours, c.fuel_reference, c.ice_reference_hours) <= 0:
            raise ValueError("Positive fixed cost reference scales required")
        self.lats, self.lons = np.asarray(latitude), np.asarray(longitude)
        self.sic, self.u, self.v = (np.asarray(x, float) for x in (sic, u, v))
        self.static = np.asarray(static, bool)
        self.icebergs = np.asarray(iceberg_intervals, bool)
        self.anchor = pd.Timestamp(anchor)
        self.shape = self.static.shape
        expected = (c.horizon_hours + 1,) + self.shape
        if any(a.shape != expected for a in (self.sic, self.u, self.v)):
            raise ValueError("Hourly environmental arrays have incorrect dimensions")
        if self.icebergs.shape != (c.horizon_hours,) + self.shape:
            raise ValueError("Iceberg interval array has incorrect dimensions")
        self.geod = Geod(ellps="WGS84")
        self.vmax = c.max_through_water_knots * KNOTS_TO_M_PER_S
        speeds = np.hypot(self.u, self.v)
        if not np.isfinite(speeds).any():
            raise ValueError("No finite current data")
        self.max_ground_kmh = (self.vmax + np.nanmax(speeds)) * 3.6
        self.geometry_cache: dict[tuple, tuple] = {}

    def _geometry(self, a: tuple[int, int], b: tuple[int, int]) -> tuple:
        key = (a, b)
        if key not in self.geometry_cache:
            r, col = a
            nr, nc = b
            az, _, metres = self.geod.inv(self.lons[col], self.lats[r], self.lons[nc], self.lats[nr])
            cells = list(dict.fromkeys([(r, col), (nr, nc), (r, nc), (nr, col)]))
            self.geometry_cache[key] = (az, metres, cells)
        return self.geometry_cache[key]

    def evaluate_edge(
        self, a: tuple[int, int], b: tuple[int, int], start_hour: int, duration: int
    ) -> dict[str, Any] | None:
        """Feasibility and cost of traversing ``a -> b`` over ``duration`` hours."""
        c = self.cfg
        stop = start_hour + duration
        if duration < 1 or start_hour < 0 or stop > c.horizon_hours:
            return None
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1:
            return None
        az, metres, cells = self._geometry(a, b)
        rr, cc = np.array(cells).T
        # All touched cells, including both diagonal side cells, must pass.
        if self.static[rr, cc].any():
            return None
        if self.icebergs[start_hour:stop, rr, cc].any():
            return None
        s = self.sic[start_hour : stop + 1, rr, cc]
        u = self.u[start_hour : stop + 1, rr, cc]
        v = self.v[start_hour : stop + 1, rr, cc]
        if not (np.isfinite(s).all() and np.isfinite(u).all() and np.isfinite(v).all()):
            return None
        if (s < 0).any() or (s > c.max_sic).any():
            return None
        # Conservative whole-leg ice value; no temporal threshold crossing omitted.
        peak_sic = float(s.max())
        allowed_speed = self.vmax * (1 - c.ice_speed_reduction * peak_sic / c.max_sic)
        ground_speed = metres / (duration * 3600.0)
        if metres:
            fractions = (np.arange(duration) + 0.5) / duration
            mlon, mlat, _ = self.geod.fwd(
                np.full(duration, self.lons[a[1]]),
                np.full(duration, self.lats[a[0]]),
                np.full(duration, az),
                fractions * metres,
            )
            bearings, _, _ = self.geod.inv(
                mlon, mlat, np.full(duration, self.lons[b[1]]), np.full(duration, self.lats[b[0]])
            )
            theta = np.deg2rad(bearings)[:, None]
            east, north = ground_speed * np.sin(theta), ground_speed * np.cos(theta)
        else:
            east = north = np.zeros((duration, 1))
        # Through-water velocity = required ground velocity minus current vector.
        # Taking worst endpoint/cell value is conservative for this interpolation model.
        required0 = np.hypot(east - u[:-1], north - v[:-1])
        required1 = np.hypot(east - u[1:], north - v[1:])
        required = np.maximum(required0, required1).max(axis=1)
        if (required > allowed_speed + 1e-8).any():
            return None
        # Dimensionless-rate integral: NOT litres, tonnes, or calibrated engine power.
        fuel = float(((required / self.vmax) ** 3 + c.hotel_proxy_per_hour).sum())
        ice_exposure = duration * peak_sic
        cost = (
            c.time_weight * duration / c.time_reference_hours
            + c.fuel_weight * fuel / c.fuel_reference
            + c.ice_weight * ice_exposure / c.ice_reference_hours
        )
        return {
            "duration_hours": duration,
            "distance_km": metres / 1000,
            "fuel_proxy": fuel,
            "sic_exposure_hours": ice_exposure,
            "max_sic": peak_sic,
            "max_required_speed_knots": float(required.max() / KNOTS_TO_M_PER_S),
            "objective": cost,
            "action": "wait" if a == b else "move",
        }

    def solve(
        self, start: tuple[int, int], goal: tuple[int, int], use_heuristic: bool = True
    ) -> dict[str, Any]:
        start, goal = tuple(start), tuple(goal)
        for name, point in [("start", start), ("goal", goal)]:
            if not (0 <= point[0] < self.shape[0] and 0 <= point[1] < self.shape[1]):
                raise ValueError(name + " outside grid")
            if self.static[point]:
                return {"status": name + "_statically_blocked"}
        if (
            self.icebergs[(0,) + start]
            or not np.isfinite(self.sic[(0,) + start])
            or self.sic[(0,) + start] > self.cfg.max_sic
        ):
            return {"status": "start_blocked_at_departure"}

        def heuristic(point: tuple[int, int]) -> float:
            # Admissible: best-case time at the maximum attainable ground speed,
            # scaled by the time term only.
            if not use_heuristic:
                return 0.0
            _, _, d = self.geod.inv(
                self.lons[point[1]], self.lats[point[0]], self.lons[goal[1]], self.lats[goal[0]]
            )
            return (d / 1000 / self.max_ground_kmh) * self.cfg.time_weight / self.cfg.time_reference_hours

        initial = (0,) + start
        best = {initial: 0.0}
        previous: dict = {}
        queue = [(heuristic(start), 0.0, initial)]
        expansions = 0
        started = time.monotonic()
        moves = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)]
        while queue:
            _, score, state = heapq.heappop(queue)
            if score > best.get(state, math.inf) + 1e-12:
                continue
            hour, r, col = state
            if (r, col) == goal:
                legs: list = []
                states = [state]
                while state != initial:
                    parent, leg = previous[state]
                    legs.append(leg)
                    states.append(parent)
                    state = parent
                states.reverse()
                legs.reverse()
                for end, leg in zip(states[1:], legs, strict=True):
                    leg["arrival_hour"] = end[0]
                return {
                    "status": "route_found",
                    "states": states,
                    "legs": legs,
                    "objective": score,
                    "expansions": expansions,
                    "runtime_seconds": time.monotonic() - started,
                }
            if (
                expansions >= self.cfg.max_expansions
                or time.monotonic() - started > self.cfg.max_runtime_seconds
            ):
                return {
                    "status": "search_limit_reached_not_proof_of_no_route",
                    "expansions": expansions,
                    "runtime_seconds": time.monotonic() - started,
                }
            expansions += 1
            for dr, dc in moves:
                nr, nc = r + dr, col + dc
                if not (0 <= nr < self.shape[0] and 0 <= nc < self.shape[1]):
                    continue
                a, b = (r, col), (nr, nc)
                if self.static[b]:
                    continue
                _, d, cells = self._geometry(a, b)
                if any(self.static[p] for p in cells):
                    continue
                durations = (
                    [1]
                    if a == b
                    else sorted(
                        {
                            max(1, int(math.ceil((d / 1000) / (speed * 1.852))))
                            for speed in self.cfg.ground_action_knots
                        }
                    )
                )
                for duration in durations:
                    leg = self.evaluate_edge(a, b, hour, duration)
                    if leg is None:
                        continue
                    newstate = (hour + duration, nr, nc)
                    newscore = score + leg["objective"]
                    if newscore < best.get(newstate, math.inf) - 1e-12:
                        best[newstate] = newscore
                        previous[newstate] = ((hour, r, col), leg)
                        heapq.heappush(queue, (newscore + heuristic(b), newscore, newstate))
        return {
            "status": "no_feasible_route_in_configured_discrete_graph",
            "expansions": expansions,
            "runtime_seconds": time.monotonic() - started,
        }


def summarise_result(planner: TimeAStar, result: dict[str, Any]) -> dict[str, Any]:
    """Waypoints, metrics and GeoJSON from a solved route (no files written).

    Same accumulation as the notebook's ``export_result``: every metric is summed
    from the legs the search chose, never from a static table.
    """
    summary: dict[str, Any] = {k: v for k, v in result.items() if k not in ("states", "legs")}
    summary["config"] = planner.cfg.as_dict()
    summary["scenario"] = "forecast_driven_demo_not_for_navigation"
    summary["limitations"] = [
        "No bathymetry/wind/waves or real vessel limits",
        "Uncalibrated buffers, ice-speed rule and cubic fuel proxy",
        "Hourly interpolation is not additional forecast information",
        "Coarse grid centres, not original requested endpoints",
        "No adaptive corridor expansion; finite region and 7-day horizon",
        "No verified all-iceberg or fragment coverage",
    ]
    if result.get("status") != "route_found":
        return summary
    rows = []
    for i, (hour, r, col) in enumerate(result["states"]):
        rows.append(
            {
                "waypoint": i,
                "elapsed_hours": int(hour),
                "arrival_time": (planner.anchor + pd.Timedelta(hours=int(hour))).isoformat(),
                "latitude": float(planner.lats[r]),
                "longitude": float(planner.lons[col]),
                "action": "departure" if i == 0 else result["legs"][i - 1]["action"],
            }
        )
    summary.update(
        waypoints=rows,
        legs=result["legs"],
        distance_km=sum(x["distance_km"] for x in result["legs"]),
        duration_hours=int(result["states"][-1][0]),
        arrival_time=rows[-1]["arrival_time"],
        fuel_proxy=sum(x["fuel_proxy"] for x in result["legs"]),
        sic_exposure_hours=sum(x["sic_exposure_hours"] for x in result["legs"]),
        geometry={
            "type": "LineString",
            "coordinates": [[r["longitude"], r["latitude"]] for r in rows],
        },
    )
    return summary
