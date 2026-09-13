"""Route planning: ports + production forecasts -> time-dependent A* -> persisted route.

This is an **integration layer**, not a model. It never retrains anything and
never re-runs the GRU or the sea-ice U-Net: it consumes the forecasts those
deployed champions already produced, assembles one coherent environment
snapshot, and hands it to :class:`ml.routing.engine.TimeAStar`.

Failure is explicit. If the sea-ice forecast, the iceberg forecast or the ocean
currents needed for the requested window are unavailable, the request fails with
a machine-readable code — no zero-SIC fallback, no fabricated currents, and
never a straight line presented as a route.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml import Forecast, ForecastSet
from app.models.routing import Port, Route
from app.models.seaice import SeaIceForecast, SeaIceForecastSet, SeaIceObservation
from app.models.tracking import Observation
from app.services.model_registry import ModelRegistry
from app.services.port_resolver import PortError, PortResolver, ResolvedPort
from app.services.seaice_registry import SeaIceRegistry
from ml.routing.config import (
    GRID_RESOLUTION_DEG,
    KNOTS_TO_M_PER_S,
    ROUTE_PLANNER_VERSION,
    RouteConfig,
)
from ml.routing.curvature import curvature_metrics
from ml.routing.engine import summarise_result
from ml.routing.environment import (
    RoutingEnvironmentError,
    build_static_exclusions,
    prepare_planner,
)
from ml.seaice.constants import HORIZONS as SEAICE_HORIZONS
from ml.seaice.constants import WINDOW as SEAICE_WINDOW
from ml.seaice.grid_geo import latitudes as seaice_latitudes
from ml.seaice.grid_geo import longitudes as seaice_longitudes
from ml.seaice.grid_store import load_forecast, load_observation

log = get_logger(__name__)
GEOD = Geod(ellps="WGS84")

# A* statuses -> API error codes.
_STATUS_ERRORS = {
    "no_feasible_route_in_configured_discrete_graph": (
        "NO_FEASIBLE_ROUTE",
        "No feasible route exists between these ports under the configured ice, "
        "iceberg and vessel constraints. Constraints were not weakened to force a route.",
    ),
    "search_limit_reached_not_proof_of_no_route": (
        "ROUTE_ENGINE_TIMEOUT",
        "The route search hit its configured runtime/expansion limit. This is not proof "
        "that no route exists.",
    ),
    "start_statically_blocked": (
        "NO_FEASIBLE_ROUTE",
        "The departure grid cell is permanently blocked (land, ice shelf or missing data).",
    ),
    "goal_statically_blocked": (
        "NO_FEASIBLE_ROUTE",
        "The destination grid cell is permanently blocked (land, ice shelf or missing data).",
    ),
    "start_blocked_at_departure": (
        "NO_FEASIBLE_ROUTE",
        "The departure cell is not navigable at departure time (sea ice above the configured "
        "maximum, or an iceberg hazard buffer).",
    ),
}


class RouteError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class GridEndpoint:
    row: int
    col: int
    latitude: float
    longitude: float
    connector_km: float
    connector_validated: bool


class RouteService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.ports = PortResolver(session, settings)

    # ------------------------------------------------------------------ ports
    def resolve_endpoints(self, departure: str | None, destination: str | None) -> tuple[ResolvedPort, ResolvedPort]:
        start = self.ports.resolve(departure, "departure")
        goal = self.ports.resolve(destination, "destination")
        if start.id == goal.id:
            raise PortError("SAME_PORT", "Departure and destination ports must be different.")
        self.ports.validate_routing_domain(start)
        self.ports.validate_routing_domain(goal)
        return start, goal

    # ------------------------------------------------- environment assembly
    def _trajectory_snapshot(self) -> dict[str, Any]:
        """The deployed trajectory champion's most recent complete forecast run."""
        champion = ModelRegistry(self.session, self.settings).get_deployed()
        if champion is None:
            raise RouteError("TRAJECTORY_FORECAST_UNAVAILABLE", "No trajectory model is deployed.")
        newest = self.session.execute(
            select(ForecastSet)
            .where(ForecastSet.model_version == champion.version)
            .order_by(ForecastSet.latest_observation_date.desc(), ForecastSet.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if newest is None:
            raise RouteError(
                "TRAJECTORY_FORECAST_UNAVAILABLE",
                f"No iceberg forecast has been generated by {champion.version}.",
            )
        anchor_date = newest.latest_observation_date
        # Only sets sharing the run's anchor date are used: mixing anchors would
        # combine forecasts made from different reference times.
        sets = list(
            self.session.execute(
                select(ForecastSet).where(
                    ForecastSet.model_version == champion.version,
                    ForecastSet.latest_observation_date == anchor_date,
                )
            ).scalars()
        )
        rows: list[dict[str, Any]] = []
        for fs in sets:
            anchor_obs = self.session.get(Observation, fs.anchor_observation_id)
            if anchor_obs is None:
                continue
            points = list(
                self.session.execute(
                    select(Forecast)
                    .where(Forecast.forecast_set_id == fs.id)
                    .order_by(Forecast.forecast_horizon_days)
                ).scalars()
            )
            if [p.forecast_horizon_days for p in points] != list(range(1, 8)):
                continue  # incomplete horizon set: skipped, never padded
            for p in points:
                rows.append(
                    {
                        "iceberg_id": fs.iceberg_id,
                        "lead_days": int(p.forecast_horizon_days),
                        "input_end_date": pd.Timestamp(anchor_date),
                        "forecast_date": pd.Timestamp(p.forecast_date),
                        "anchor_latitude": float(anchor_obs.latitude),
                        "anchor_longitude": float(anchor_obs.longitude),
                        "predicted_latitude": float(p.predicted_latitude),
                        "predicted_longitude": float(p.predicted_longitude),
                    }
                )
        if not rows:
            raise RouteError(
                "TRAJECTORY_FORECAST_UNAVAILABLE",
                "No complete D+1..D+7 iceberg forecast is available for the current anchor.",
            )
        return {
            "model_version": champion.version,
            "forecast_run_id": newest.forecast_run_id,
            "anchor_date": anchor_date,
            "frame": pd.DataFrame(rows),
            "icebergs": len({r["iceberg_id"] for r in rows}),
        }

    def _seaice_snapshot(self, anchor_date: Any) -> dict[str, Any]:
        """The deployed sea-ice champion's forecast anchored on the same date."""
        champion = SeaIceRegistry(self.session, self.settings).get_champion()
        if champion is None:
            raise RouteError("SEA_ICE_FORECAST_UNAVAILABLE", "No sea-ice model is deployed.")
        forecast_set = self.session.execute(
            select(SeaIceForecastSet)
            .join(SeaIceObservation, SeaIceForecastSet.anchor_observation_id == SeaIceObservation.id)
            .where(
                SeaIceForecastSet.model_version == champion.version,
                SeaIceForecastSet.anchor_date == anchor_date,
                SeaIceObservation.n_valid_cells > 0,
            )
            .order_by(SeaIceForecastSet.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if forecast_set is None:
            # The iceberg feed is weekly and the sea-ice feed daily, so a forecast
            # anchored on the trajectory reference time may not exist yet. Produce
            # it by running the deployed champion's INFERENCE on the window ending
            # at that date — no training, and it is persisted so later routes reuse
            # it instead of recomputing.
            from app.services.seaice_forecast_service import SeaIceForecastService

            outcome = SeaIceForecastService(self.session, self.settings).run(
                trigger="route_planning", anchor_date=anchor_date
            )
            forecast_set = self.session.execute(
                select(SeaIceForecastSet)
                .where(
                    SeaIceForecastSet.model_version == champion.version,
                    SeaIceForecastSet.anchor_date == anchor_date,
                )
                .order_by(SeaIceForecastSet.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if forecast_set is None:
                raise RouteError(
                    "SEA_ICE_FORECAST_UNAVAILABLE",
                    f"No sea-ice forecast could be produced for the forecast reference time "
                    f"{anchor_date}: {outcome.error or outcome.status}.",
                )
        points = list(
            self.session.execute(
                select(SeaIceForecast)
                .where(SeaIceForecast.forecast_set_id == forecast_set.id)
                .order_by(SeaIceForecast.horizon_days)
            ).scalars()
        )
        if [p.horizon_days for p in points] != list(SEAICE_HORIZONS):
            raise RouteError(
                "SEA_ICE_FORECAST_UNAVAILABLE",
                f"Sea-ice forecast is incomplete; expected horizons {list(SEAICE_HORIZONS)}.",
            )
        anchor_observation = self.session.get(SeaIceObservation, forecast_set.anchor_observation_id)
        window = list(
            self.session.execute(
                select(SeaIceObservation)
                .where(
                    SeaIceObservation.observation_date <= anchor_observation.observation_date,
                    SeaIceObservation.n_valid_cells > 0,
                )
                .order_by(SeaIceObservation.observation_date.desc())
                .limit(SEAICE_WINDOW)
            ).scalars()
        )
        return {
            "model_version": champion.version,
            "forecast_run_id": forecast_set.run_id,
            "forecast_set": forecast_set,
            "anchor_observation": anchor_observation,
            "window": window[::-1],
            "forecasts": points,
        }

    def _currents(self, anchor: pd.Timestamp, horizon_hours: int) -> dict[str, Any]:
        """Copernicus Marine surface currents covering the planning horizon.

        Uses the configured provider spec (the same catalogue entry the
        environmental pipeline uses). Missing coverage is an explicit failure:
        currents are never assumed to be zero.
        """
        from ml.environment.providers.copernicus_marine import SPECS, _credentials

        spec = SPECS["copernicus_marine_current_anfc"]
        username, password = _credentials(
            self.settings.copernicus_marine_username.get_secret_value()
            if self.settings.copernicus_marine_username
            else None,
            self.settings.copernicus_marine_password.get_secret_value()
            if self.settings.copernicus_marine_password
            else None,
        )
        if not (username and password):
            raise RouteError(
                "ENVIRONMENT_DATA_UNAVAILABLE",
                "Ocean-current credentials are not configured; routing cannot proceed without currents.",
            )
        s = self.settings
        end = anchor + pd.Timedelta(hours=horizon_hours)
        cache_dir = Path(s.seaice_data_dir).parent / "routing" / "currents"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = (
            f"{spec.dataset_id}_{anchor.strftime('%Y%m%d')}_{horizon_hours}h_"
            f"{s.routing_domain_west:.0f}_{s.routing_domain_east:.0f}_"
            f"{s.routing_domain_south:.0f}_{s.routing_domain_north:.0f}.nc"
        )
        cached = cache_dir / key
        if cached.exists():
            import xarray as xr

            try:
                dataset = xr.open_dataset(cached).load()
            except Exception as exc:  # noqa: BLE001 - a truncated/corrupt cache must not fail the route
                log.warning("route_currents_cache_unreadable", path=str(cached), error=str(exc))
                cached.unlink(missing_ok=True)
            else:
                log.info("route_currents_cache_hit", path=str(cached))
                return {"dataset": dataset, "spec": spec}
        try:
            import copernicusmarine

            dataset = copernicusmarine.open_dataset(
                dataset_id=spec.dataset_id,
                variables=["uo", "vo"],
                minimum_longitude=s.routing_domain_west - 1,
                maximum_longitude=s.routing_domain_east + 1,
                minimum_latitude=s.routing_domain_south - 1,
                maximum_latitude=s.routing_domain_north + 1,
                start_datetime=(anchor - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00"),
                end_datetime=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT23:59:59"),
                username=username,
                password=password,
            )
        except Exception as exc:  # noqa: BLE001 - external service
            raise RouteError(
                "ENVIRONMENT_DATA_UNAVAILABLE",
                f"Ocean-current data could not be retrieved ({type(exc).__name__}).",
            ) from exc
        if "depth" in dataset.coords:
            surface = int(np.nanargmin(dataset["depth"].values))
            dataset = dataset.isel(depth=surface, drop=True)
        for name in ("uo", "vo"):
            units = dataset[name].attrs.get("units")
            if units not in ("m s-1", "m/s", "m s**-1"):
                raise RouteError(
                    "ENVIRONMENT_DATA_UNAVAILABLE",
                    f"Unexpected current units for {name}: {units!r}; metres/second required.",
                )
        loaded = dataset[["uo", "vo"]].transpose("time", "latitude", "longitude").load()
        # Write then rename: a crash mid-write must never leave a half-file that
        # a later request would try to read.
        partial = cached.with_suffix(".nc.partial")
        loaded.to_netcdf(partial)
        partial.replace(cached)
        log.info("route_currents_cached", path=str(cached), sizes=dict(loaded.sizes))
        return {"dataset": loaded, "spec": spec}

    # -------------------------------------------------------- grid endpoints
    def _grid_endpoint(
        self, latitudes: np.ndarray, longitudes: np.ndarray, blocked: np.ndarray, port: ResolvedPort
    ) -> GridEndpoint:
        """Map a port to a navigable grid cell and validate the connector.

        The port's own cell is used when navigable. Otherwise the nearest
        navigable cell within ``ROUTING_MAX_CONNECTOR_KM`` is used, and the
        connector is only accepted if the straight line from the port to that
        cell does not cross a blocked cell — the notebook's saved run recorded
        connectors as *not* validated; here they are.
        """
        row = int(np.argmin(np.abs(latitudes - port.latitude)))
        col = int(np.argmin(np.abs(longitudes - port.longitude)))
        if not blocked[row, col]:
            _, _, metres = GEOD.inv(port.longitude, port.latitude, float(longitudes[col]), float(latitudes[row]))
            return GridEndpoint(row, col, float(latitudes[row]), float(longitudes[col]), metres / 1000, True)

        rows, cols = np.nonzero(~blocked)
        if rows.size == 0:
            raise RouteError("ROUTING_GRID_UNAVAILABLE", "No navigable cell exists in the routing grid.")
        _, _, metres = GEOD.inv(
            np.full(rows.shape, port.longitude),
            np.full(rows.shape, port.latitude),
            longitudes[cols],
            latitudes[rows],
        )
        order = np.argsort(metres)
        for index in order:
            distance_km = float(metres[index]) / 1000
            if distance_km > self.settings.routing_max_connector_km:
                break
            r, c = int(rows[index]), int(cols[index])
            if self._connector_is_clear(latitudes, longitudes, blocked, port, r, c):
                return GridEndpoint(r, c, float(latitudes[r]), float(longitudes[c]), distance_km, True)
        raise RouteError(
            "NO_FEASIBLE_ROUTE",
            f"{port.name} could not be connected to a navigable routing cell within "
            f"{self.settings.routing_max_connector_km:.0f} km without crossing blocked water.",
        )

    @staticmethod
    def _connector_is_clear(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        blocked: np.ndarray,
        port: ResolvedPort,
        row: int,
        col: int,
    ) -> bool:
        """Sample the port -> cell segment; every cell it crosses must be navigable.

        The port's OWN cell is exempt. At 0.5 degrees a cell containing a harbour
        always intersects mapped land, so it is always "blocked" for transit
        purposes — that is the cell the vessel is berthed in, not water it has to
        cross. Every other cell along the connector must be navigable, which is
        what makes the connector validated rather than assumed.
        """
        target_lat, target_lon = float(latitudes[row]), float(longitudes[col])
        origin_row = int(np.argmin(np.abs(latitudes - port.latitude)))
        origin_col = int(np.argmin(np.abs(longitudes - port.longitude)))
        _, _, metres = GEOD.inv(port.longitude, port.latitude, target_lon, target_lat)
        steps = max(2, int(metres / 5000) + 1)  # ~5 km sampling
        lons = np.linspace(port.longitude, target_lon, steps)
        lats = np.linspace(port.latitude, target_lat, steps)
        for lon, lat in zip(lons, lats, strict=True):
            r = int(np.argmin(np.abs(latitudes - lat)))
            c = int(np.argmin(np.abs(longitudes - lon)))
            if (r, c) == (origin_row, origin_col):
                continue
            if blocked[r, c]:
                return False
        return True


    @staticmethod
    def _corridor_conditions(
        planner: Any, start_cell: GridEndpoint, goal_cell: GridEndpoint, geodesic_km: float
    ) -> dict[str, Any]:
        """Ice statistics along the straight corridor, for explaining a failure.

        Sampled from the hourly field the search used. The ice-reduced speed is
        the planner's own rule applied to the corridor's peak concentration.
        """
        rows = np.linspace(start_cell.row, goal_cell.row, 40).astype(int)
        cols = np.linspace(start_cell.col, goal_cell.col, 40).astype(int)
        samples = planner.sic[:, rows, cols]
        if not np.isfinite(samples).any():
            return {"geodesic_km": geodesic_km, "mean_sic": None}
        mean_sic = float(np.nanmean(samples))
        max_sic = float(np.nanmax(samples))
        config = planner.cfg
        allowed_m_s = planner.vmax * (1 - config.ice_speed_reduction * min(max_sic, config.max_sic) / config.max_sic)
        allowed_knots = max(allowed_m_s / KNOTS_TO_M_PER_S, 0.1)
        usable_knots = min(max(config.ground_action_knots), allowed_knots)
        return {
            "geodesic_km": geodesic_km,
            "mean_sic": mean_sic,
            "max_sic": max_sic,
            "ice_reduced_speed_knots": usable_knots,
            "hours_at_ice_reduced_speed": geodesic_km / (usable_knots * 1.852),
            "horizon_hours": config.horizon_hours,
        }

    # -------------------------------------------------------------- planning
    def plan(
        self, departure: str | None, destination: str | None, max_sic: float | None = None
    ) -> Route:
        """``max_sic`` is the vessel's ice capability, not a physical constant.

        The notebook calls its 0.30 an "illustrative software-test configuration,
        NOT a real vessel rating", so it is a per-request vessel parameter that
        defaults to that value. The constraint is enforced exactly as before —
        a higher value means a more ice-capable vessel, never a relaxed search.
        """
        start_port, goal_port = self.resolve_endpoints(departure, destination)
        route = Route(
            route_id=str(uuid.uuid4()),
            status="RUNNING",
            departure_port_id=start_port.id,
            destination_port_id=goal_port.id,
            requested_departure_latitude=start_port.latitude,
            requested_departure_longitude=start_port.longitude,
            requested_destination_latitude=goal_port.latitude,
            requested_destination_longitude=goal_port.longitude,
            route_planner_version=ROUTE_PLANNER_VERSION,
        )
        self.session.add(route)
        self.session.flush()
        try:
            self._compute(route, start_port, goal_port, max_sic)
        except (RouteError, RoutingEnvironmentError) as exc:
            route.status = "NO_FEASIBLE_ROUTE" if getattr(exc, "code", "") == "NO_FEASIBLE_ROUTE" else "FAILED"
            route.error_code = exc.code
            route.error_message = str(exc)
            self.session.flush()
            log.warning("route_failed", route_id=route.route_id, code=exc.code, error=str(exc))
        return route

    def _compute(
        self, route: Route, start_port: ResolvedPort, goal_port: ResolvedPort, max_sic: float | None
    ) -> None:
        defaults = RouteConfig()
        config = RouteConfig(
            max_runtime_seconds=self.settings.route_max_runtime_seconds,
            max_expansions=self.settings.route_max_expansions,
            max_sic=defaults.max_sic if max_sic is None else float(max_sic),
        )
        trajectory = self._trajectory_snapshot()
        anchor_date = trajectory["anchor_date"]
        seaice = self._seaice_snapshot(anchor_date)
        anchor = pd.Timestamp(anchor_date)

        # --- grid: the routing domain subset of the sea-ice grid -------------
        all_lats, all_lons = seaice_latitudes(), seaice_longitudes()
        s = self.settings
        lat_mask = (all_lats >= s.routing_domain_south) & (all_lats <= s.routing_domain_north)
        lon_mask = (all_lons >= s.routing_domain_west) & (all_lons <= s.routing_domain_east)
        latitudes, longitudes = all_lats[lat_mask], all_lons[lon_mask]
        if latitudes.size < 2 or longitudes.size < 2:
            raise RouteError("ROUTING_GRID_UNAVAILABLE", "The routing domain does not intersect the sea-ice grid.")
        rows_idx = np.nonzero(lat_mask)[0]
        cols_idx = np.nonzero(lon_mask)[0]

        def subset(grid: np.ndarray) -> np.ndarray:
            return grid[np.ix_(rows_idx, cols_idx)]

        observed, observed_mask = load_observation(Path(seaice["anchor_observation"].grid_path))
        sic_observed = subset(observed)
        sic_forecast = np.stack([subset(load_forecast(Path(p.grid_path))) for p in seaice["forecasts"]])
        lead_hours = np.array([h * 24 for h in SEAICE_HORIZONS], dtype=float)

        # Coverage: our stored grids carry a per-cell validity mask rather than
        # the notebook's fraction of valid native pixels, so "at least 95% valid
        # on EACH input day" is applied as "valid on every day of the window".
        window_masks = [subset(load_observation(Path(o.grid_path))[1]) for o in seaice["window"]]
        history_valid_fraction = np.min(np.stack([*window_masks, observed_mask[np.ix_(rows_idx, cols_idx)]]), axis=0)

        exclusions = build_static_exclusions(
            latitudes, longitudes, history_valid_fraction, sic_forecast, GRID_RESOLUTION_DEG
        )

        # --- endpoints and the horizon feasibility pre-check ------------------
        start_cell = self._grid_endpoint(latitudes, longitudes, exclusions.blocked, start_port)
        goal_cell = self._grid_endpoint(latitudes, longitudes, exclusions.blocked, goal_port)
        _, _, direct_m = GEOD.inv(
            start_cell.longitude, start_cell.latitude, goal_cell.longitude, goal_cell.latitude
        )
        fastest_hours = (direct_m / 1000) / (max(config.ground_action_knots) * 1.852)
        if fastest_hours > config.horizon_hours:
            raise RouteError(
                "ROUTE_HORIZON_EXCEEDED",
                f"Even at the best modelled speed this voyage needs about {fastest_hours:.0f} h, "
                f"beyond the {config.horizon_hours} h forecast horizon. Long-range routes are not "
                "supported because the forecasts that constrain them do not extend that far.",
            )

        currents = self._currents(anchor, config.horizon_hours)
        dataset = currents["dataset"]
        gridded = dataset.interp(
            latitude=("latitude", latitudes), longitude=("longitude", longitudes), method="linear"
        )
        planner = prepare_planner(
            latitudes=latitudes,
            longitudes=longitudes,
            sic_observed=sic_observed,
            sic_forecast=sic_forecast,
            sic_forecast_lead_hours=lead_hours,
            current_times=pd.DatetimeIndex(gridded.time.values),
            current_u=gridded["uo"].values,
            current_v=gridded["vo"].values,
            iceberg_forecasts=trajectory["frame"],
            unresolved_hazards=pd.DataFrame(columns=["date", "latitude", "longitude"]),
            static_blocked=exclusions.blocked,
            anchor=anchor,
            config=config,
        )
        result = planner.solve((start_cell.row, start_cell.col), (goal_cell.row, goal_cell.col))
        summary = summarise_result(planner, result)

        route.trajectory_model_version = trajectory["model_version"]
        route.trajectory_forecast_run_id = trajectory["forecast_run_id"]
        route.sea_ice_model_version = seaice["model_version"]
        route.sea_ice_forecast_run_id = seaice["forecast_run_id"]
        route.forecast_reference_time = anchor.tz_localize(UTC) if anchor.tzinfo is None else anchor
        route.resolved_departure_latitude = start_cell.latitude
        route.resolved_departure_longitude = start_cell.longitude
        route.resolved_destination_latitude = goal_cell.latitude
        route.resolved_destination_longitude = goal_cell.longitude
        route.departure_connector_km = start_cell.connector_km
        route.destination_connector_km = goal_cell.connector_km
        route.connectors_validated = start_cell.connector_validated and goal_cell.connector_validated
        route.expansions = summary.get("expansions")
        route.runtime_seconds = summary.get("runtime_seconds")
        route.config = config.as_dict()
        route.environment_snapshot = {
            "forecast_reference_time": anchor.isoformat(),
            "trajectory_model_version": trajectory["model_version"],
            "trajectory_forecast_run_id": trajectory["forecast_run_id"],
            "trajectory_icebergs": trajectory["icebergs"],
            "sea_ice_model_version": seaice["model_version"],
            "sea_ice_forecast_run_id": seaice["forecast_run_id"],
            "sea_ice_horizons_days": list(SEAICE_HORIZONS),
            "current_dataset_id": currents["spec"].dataset_id,
            "current_provider": currents["spec"].name,
            "route_planner_version": ROUTE_PLANNER_VERSION,
            "grid": {
                "resolution_deg": GRID_RESOLUTION_DEG,
                "shape": [int(latitudes.size), int(longitudes.size)],
                "domain": [s.routing_domain_west, s.routing_domain_east, s.routing_domain_south, s.routing_domain_north],
                "exclusions": exclusions.counts(),
            },
            "vessel_max_sic": config.max_sic,
            "search_status": summary.get("status"),
        }

        if result.get("status") != "route_found":
            code, message = _STATUS_ERRORS.get(
                result["status"], ("ROUTE_ENGINE_ERROR", f"Route engine returned {result['status']}.")
            )
            # Explain WHY, from the environment the search actually saw, rather
            # than leaving an opaque failure. These are observations, not excuses:
            # the constraints are never relaxed on the strength of them.
            corridor = self._corridor_conditions(planner, start_cell, goal_cell, direct_m / 1000)
            route.environment_snapshot = {**(route.environment_snapshot or {}), "corridor": corridor}
            if code == "NO_FEASIBLE_ROUTE" and corridor.get("mean_sic") is not None:
                message += (
                    f" Corridor sea ice averages {corridor['mean_sic']:.0%} (peak {corridor['max_sic']:.0%}) over the "
                    f"{config.horizon_hours} h horizon; at the resulting ice-reduced speed the "
                    f"{corridor['geodesic_km']:.0f} km crossing needs about {corridor['hours_at_ice_reduced_speed']:.0f} h, "
                    f"against a {config.horizon_hours} h forecast horizon."
                )
            raise RouteError(code, message)

        stats, turns = curvature_metrics(
            summary["waypoints"],
            request={
                "start": (start_port.latitude, start_port.longitude),
                "destination": (goal_port.latitude, goal_port.longitude),
                "connectors_validated": route.connectors_validated,
            },
        )
        # One vessel at a time: the previous voyage is completed, not deleted, so
        # route history and its model provenance stay intact.
        for previous in self.session.execute(
            select(Route).where(Route.status == "ACTIVE", Route.id != route.id)
        ).scalars():
            previous.status = "COMPLETED"
            previous.completed_at = datetime.now(UTC)
        route.status = "ACTIVE"  # a successful computation becomes the active voyage
        route.distance_km = summary["distance_km"]
        route.duration_hours = float(summary["duration_hours"])
        route.fuel_proxy = summary["fuel_proxy"]
        route.sic_exposure_hours = summary["sic_exposure_hours"]
        route.weighted_objective = summary["objective"]
        route.confidence_status = "not_calibrated"
        route.curvature = {**stats, "turns": turns}
        route.waypoints = summary["waypoints"]
        coordinates = ", ".join(f"{w['longitude']:.6f} {w['latitude']:.6f}" for w in summary["waypoints"])
        route.geom = f"SRID=4326;LINESTRING({coordinates})"
        self.session.flush()
        log.info(
            "route_found",
            route_id=route.route_id,
            distance_km=round(route.distance_km, 1),
            duration_hours=route.duration_hours,
            expansions=route.expansions,
        )

    # ----------------------------------------------------------------- reads
    def get(self, route_id: str) -> Route | None:
        return self.session.execute(select(Route).where(Route.route_id == route_id)).scalar_one_or_none()

    def active(self) -> Route | None:
        return self.session.execute(
            select(Route).where(Route.status == "ACTIVE").order_by(Route.created_at.desc(), Route.id.desc()).limit(1)
        ).scalar_one_or_none()

    def history(self, limit: int = 20) -> list[Route]:
        return list(
            self.session.execute(
                select(Route).order_by(Route.created_at.desc(), Route.id.desc()).limit(limit)
            ).scalars()
        )

    def port(self, port_id: int) -> Port | None:
        return self.session.get(Port, port_id)


def estimated_arrival(route: Route) -> datetime | None:
    if route.forecast_reference_time is None or route.duration_hours is None:
        return None
    return route.forecast_reference_time + timedelta(hours=route.duration_hours)
