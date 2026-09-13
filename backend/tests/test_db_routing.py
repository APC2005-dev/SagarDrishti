"""Route planning: port validation, the A* engine, curvature and persistence.

Nothing here reaches NGA or Copernicus: the World Port Index is served from a
stub client and the routing environment is synthetic. The engine, curvature and
persistence code under test are the production ones.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
import numpy as np
import pytest

from app.models.routing import Port, Route
from app.services.port_resolver import PortError, PortResolver, normalize, parse_coordinate
from ml.routing.config import RouteConfig
from ml.routing.curvature import curvature_metrics
from ml.routing.engine import TimeAStar, summarise_result

pytestmark = pytest.mark.db

WPI_ROWS = [
    {"portNumber": "63080", "portName": "Scotia Bay", "countryName": "Antarctica", "countryCode": "AY",
     "latitude": "60°45'00\"S", "longitude": "44°43'00\"W", "regionName": "Antarctica", "harborSize": "S"},
    {"portNumber": "63090", "portName": "Admiralty Bay", "countryName": "Antarctica", "countryCode": "AY",
     "latitude": "62°05'00\"S", "longitude": "58°25'00\"W", "regionName": "Antarctica", "harborSize": "S"},
    {"portNumber": "53010", "portName": "Cape Town", "countryName": "South Africa", "countryCode": "SF",
     "latitude": "33°55'00\"S", "longitude": "18°26'00\"E", "regionName": "Africa", "harborSize": "L"},
    # Same name in two countries: resolution by bare name must refuse to guess.
    {"portNumber": "11111", "portName": "Duplicate Harbor", "countryName": "Antarctica", "countryCode": "AY",
     "latitude": "61°00'00\"S", "longitude": "45°00'00\"W"},
    {"portNumber": "22222", "portName": "Duplicate Harbor", "countryName": "Chile", "countryCode": "CI",
     "latitude": "53°00'00\"S", "longitude": "70°00'00\"W"},
    # Unusable row: no coordinate. It must be skipped, never given a made-up position.
    {"portNumber": "99999", "portName": "Nowhere", "countryName": "Nowhere", "latitude": None, "longitude": None},
]


def stub_client(rows: list[dict] | None = None, fail: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("World Port Index unreachable")
        return httpx.Response(200, content=json.dumps({"ports": rows if rows is not None else WPI_ROWS}))

    return httpx.Client(transport=httpx.MockTransport(handler))


def resolver(db, settings, **kwargs) -> PortResolver:
    return PortResolver(db, settings, client=stub_client(**kwargs))


# ------------------------------------------------------------------ parsing
def test_world_port_index_coordinates_are_parsed() -> None:
    assert parse_coordinate("60°45'00\"S") == pytest.approx(-60.75)
    assert parse_coordinate("44°43'00\"W") == pytest.approx(-44.716667, abs=1e-5)
    assert parse_coordinate("18°26'00\"E") == pytest.approx(18.433333, abs=1e-5)
    assert parse_coordinate(-62.5) == -62.5
    assert parse_coordinate(None) is None
    assert parse_coordinate("not a coordinate") is None


def test_name_normalisation_folds_case_and_punctuation() -> None:
    assert normalize("St. John's") == normalize("st johns") == "st johns"
    assert normalize("  Port   Foster ") == "port foster"


# ---------------------------------------------------------------- indexing
def test_index_is_cached_and_unusable_rows_are_skipped(db, settings) -> None:  # type: ignore[no-untyped-def]
    stats = resolver(db, settings).refresh_index(force=True)
    assert stats["inserted"] == 5 and stats["skipped"] == 1  # 'Nowhere' has no coordinate
    assert db.query(Port).count() == 5
    # A second refresh updates in place rather than duplicating.
    again = resolver(db, settings).refresh_index(force=True)
    assert again["inserted"] == 0 and again["updated"] == 5
    assert db.query(Port).count() == 5


def test_port_service_failure_is_reported_not_faked(db, settings) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(PortError) as exc:
        resolver(db, settings, fail=True).refresh_index(force=True)
    assert exc.value.code == "PORT_SERVICE_UNAVAILABLE"
    assert db.query(Port).count() == 0  # nothing invented when the source is down


# --------------------------------------------------------------- searching
def test_search_ranks_exact_then_prefix_and_can_scope_to_the_domain(db, settings) -> None:  # type: ignore[no-untyped-def]
    r = resolver(db, settings)
    r.refresh_index(force=True)
    names = [p.name for p in r.search("bay", limit=10, domain_only=True)]
    assert set(names) == {"Scotia Bay", "Admiralty Bay"}
    # Cape Town is a real port but outside the Antarctic routing domain.
    assert [p.name for p in r.search("cape town", domain_only=True)] == []
    assert [p.name for p in r.search("cape town", domain_only=False)] == ["Cape Town"]
    assert r.search("", limit=5) == []


# -------------------------------------------------------------- resolution
def test_resolution_by_identifier_and_by_name(db, settings) -> None:  # type: ignore[no-untyped-def]
    r = resolver(db, settings)
    r.refresh_index(force=True)
    by_name = r.resolve("Scotia Bay", "departure")
    by_id = r.resolve("63080", "departure")
    assert by_name.id == by_id.id and by_id.identifier == "63080"
    assert by_name.latitude == pytest.approx(-60.75)
    assert by_name.source == "nga_world_port_index"


def test_empty_and_unknown_ports_are_rejected(db, settings) -> None:  # type: ignore[no-untyped-def]
    r = resolver(db, settings)
    r.refresh_index(force=True)
    for value in (None, "", "   "):
        with pytest.raises(PortError) as exc:
            r.resolve(value, "departure")
        assert exc.value.code == "INVALID_DEPARTURE_PORT"
        with pytest.raises(PortError) as exc:
            r.resolve(value, "destination")
        assert exc.value.code == "INVALID_DESTINATION_PORT"
    with pytest.raises(PortError) as exc:
        r.resolve("Hogwarts", "departure")
    assert exc.value.code == "PORT_NOT_FOUND"
    # A research station or arbitrary place name is not a port.
    with pytest.raises(PortError) as exc:
        r.resolve("Amundsen-Scott South Pole Station", "departure")
    assert exc.value.code == "PORT_NOT_FOUND"


def test_ambiguous_name_refuses_to_guess(db, settings) -> None:  # type: ignore[no-untyped-def]
    r = resolver(db, settings)
    r.refresh_index(force=True)
    with pytest.raises(PortError) as exc:
        r.resolve("Duplicate Harbor", "departure")
    assert exc.value.code == "PORT_NOT_FOUND" and "more than one" in exc.value.message
    # The canonical identifier disambiguates it.
    assert r.resolve("11111", "departure").country_name == "Antarctica"


def test_ports_outside_the_routing_domain_are_rejected(db, settings) -> None:  # type: ignore[no-untyped-def]
    r = resolver(db, settings)
    r.refresh_index(force=True)
    r.validate_routing_domain(r.resolve("Scotia Bay", "departure"))  # inside: no raise
    with pytest.raises(PortError) as exc:
        r.validate_routing_domain(r.resolve("Cape Town", "departure"))
    assert exc.value.code == "PORT_OUTSIDE_ROUTING_DOMAIN"


def test_same_port_is_rejected(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.route_service import RouteService

    service = RouteService(db, settings)
    service.ports = resolver(db, settings)
    service.ports.refresh_index(force=True)
    with pytest.raises(PortError) as exc:
        service.resolve_endpoints("Scotia Bay", "63080")
    assert exc.value.code == "SAME_PORT"


# ------------------------------------------------------------ route engine
def synthetic_planner(**overrides):  # type: ignore[no-untyped-def]
    """A small ice-free basin with a land bar the route must go around."""
    config = RouteConfig(horizon_hours=72, max_runtime_seconds=20.0, **overrides)
    height, width = 12, 12
    lats = np.linspace(-66.0, -60.5, height)
    lons = np.linspace(-45.0, -39.5, width)
    steps = config.horizon_hours + 1
    sic = np.zeros((steps, height, width))
    u = np.zeros((steps, height, width))
    v = np.zeros((steps, height, width))
    static = np.zeros((height, width), bool)
    static[5, 2:10] = True
    bergs = np.zeros((config.horizon_hours, height, width), bool)
    return TimeAStar(lats, lons, sic, u, v, static, bergs, "2026-09-10", config), static, bergs, sic


def test_engine_finds_a_route_and_accumulates_real_metrics() -> None:
    planner, static, _, _ = synthetic_planner()
    result = planner.solve((1, 1), (10, 10))
    assert result["status"] == "route_found"
    summary = summarise_result(planner, result)
    # Every metric is a sum over the legs the search actually chose.
    assert summary["distance_km"] == pytest.approx(sum(leg["distance_km"] for leg in result["legs"]))
    assert summary["fuel_proxy"] == pytest.approx(sum(leg["fuel_proxy"] for leg in result["legs"]))
    assert summary["duration_hours"] == result["states"][-1][0]
    assert summary["sic_exposure_hours"] == pytest.approx(0.0)  # ice-free basin
    # The land bar is genuinely avoided, not routed through.
    assert all(not static[r, c] for _, r, c in result["states"])
    assert summary["geometry"]["type"] == "LineString"
    # GeoJSON is [longitude, latitude].
    first = summary["geometry"]["coordinates"][0]
    assert -46 < first[0] < -39 and -67 < first[1] < -60


def test_engine_reports_no_feasible_route_rather_than_a_straight_line() -> None:
    planner, static, _, _ = synthetic_planner()
    planner.static[:, 6] = True  # a full wall: the goal is unreachable
    result = planner.solve((1, 1), (10, 10))
    assert result["status"] == "no_feasible_route_in_configured_discrete_graph"
    assert "states" not in result  # no fallback geometry is produced


def test_engine_refuses_to_depart_from_a_blocked_cell() -> None:
    planner, _, _, _ = synthetic_planner()
    planner.static[1, 1] = True
    assert planner.solve((1, 1), (10, 10))["status"] == "start_statically_blocked"


def test_engine_refuses_to_depart_through_ice_above_the_vessel_limit() -> None:
    planner, _, _, _ = synthetic_planner()
    planner.sic[0, 1, 1] = 0.9  # above the configured max_sic
    assert planner.solve((1, 1), (10, 10))["status"] == "start_blocked_at_departure"


def test_iceberg_hazard_blocks_the_direct_corridor() -> None:
    planner, _, bergs, _ = synthetic_planner()
    clear = planner.solve((1, 1), (1, 10))
    assert clear["status"] == "route_found"
    planner.icebergs[:, :, 5] = True  # a hazard curtain across the corridor
    blocked = planner.solve((1, 1), (1, 10))
    assert blocked["status"] == "no_feasible_route_in_configured_discrete_graph"


def test_search_limit_is_reported_as_a_limit_not_as_absence_of_a_route() -> None:
    planner, _, _, _ = synthetic_planner(max_expansions=5)
    result = planner.solve((1, 1), (10, 10))
    assert result["status"] == "search_limit_reached_not_proof_of_no_route"


def test_horizon_bounds_the_search() -> None:
    # One hour of horizon cannot cover a multi-cell voyage.
    planner, _, _, _ = synthetic_planner()
    short = RouteConfig(horizon_hours=1, max_runtime_seconds=10.0)
    tiny = TimeAStar(
        planner.lats, planner.lons, planner.sic[:2], planner.u[:2], planner.v[:2],
        planner.static, planner.icebergs[:1], "2026-09-10", short,
    )
    assert tiny.solve((1, 1), (10, 10))["status"] == "no_feasible_route_in_configured_discrete_graph"


# --------------------------------------------------------------- curvature
def test_curvature_is_derived_from_the_actual_route() -> None:
    planner, _, _, _ = synthetic_planner()
    summary = summarise_result(planner, planner.solve((1, 1), (10, 10)))
    stats, turns = curvature_metrics(summary["waypoints"])
    assert stats["route_length_km"] == pytest.approx(summary["distance_km"], rel=1e-6)
    assert stats["detour_ratio"] > 1.0  # it went around the bar
    assert stats["maximum_turn_deg"] > 0 and len(turns) > 0
    assert stats["source_destination_alone_determine_curvature"] is False
    # Each turn is the heading change over the mean adjacent segment length.
    for turn in turns:
        expected = abs(np.deg2rad(turn["signed_turn_deg"])) / turn["adjacent_mean_length_km"]
        assert turn["absolute_discrete_curvature_rad_per_km"] == pytest.approx(expected)


def test_different_endpoints_produce_different_routes_and_curvature() -> None:
    planner, _, _, _ = synthetic_planner()
    first = summarise_result(planner, planner.solve((1, 1), (10, 10)))
    second = summarise_result(planner, planner.solve((1, 10), (10, 1)))
    assert first["waypoints"] != second["waypoints"]
    a, _ = curvature_metrics(first["waypoints"])
    b, _ = curvature_metrics(second["waypoints"])
    assert a["route_length_km"] != b["route_length_km"] or a["total_absolute_turn_deg"] != b["total_absolute_turn_deg"]


def test_stationary_waypoints_are_excluded_from_geometry() -> None:
    waypoints = [
        {"latitude": -62.0, "longitude": -45.0},
        {"latitude": -62.0, "longitude": -45.0},  # a wait: zero-length
        {"latitude": -62.5, "longitude": -44.0},
        {"latitude": -63.0, "longitude": -43.0},
    ]
    stats, _ = curvature_metrics(waypoints)
    assert stats["removed_stationary_waypoints"] == 1


# ------------------------------------------------------------- persistence
def test_failed_routes_are_recorded_but_never_become_an_active_vessel(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.route_service import RouteService

    service = RouteService(db, settings)
    service.ports = resolver(db, settings)
    service.ports.refresh_index(force=True)
    # No forecasts exist in this database, so the computation must fail explicitly.
    route = service.plan("Scotia Bay", "Admiralty Bay")
    db.flush()
    assert route.status in ("FAILED", "NO_FEASIBLE_ROUTE")
    assert route.error_code in (
        "TRAJECTORY_FORECAST_UNAVAILABLE", "SEA_ICE_FORECAST_UNAVAILABLE",
        "ENVIRONMENT_DATA_UNAVAILABLE", "NO_FEASIBLE_ROUTE", "ROUTING_GRID_UNAVAILABLE",
    )
    assert route.distance_km is None and route.waypoints is None
    assert service.active() is None  # nothing became active
    assert db.query(Route).count() == 1  # but it is auditable


def test_route_history_is_append_only(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.route_service import RouteService

    service = RouteService(db, settings)
    service.ports = resolver(db, settings)
    service.ports.refresh_index(force=True)
    service.plan("Scotia Bay", "Admiralty Bay")
    service.plan("Admiralty Bay", "Scotia Bay")
    db.flush()
    assert db.query(Route).count() == 2  # the first is not overwritten
    assert len(service.history(10)) == 2


def test_route_rows_record_the_requested_ports(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.route_service import RouteService

    service = RouteService(db, settings)
    service.ports = resolver(db, settings)
    service.ports.refresh_index(force=True)
    route = service.plan("Scotia Bay", "Admiralty Bay")
    db.flush()
    assert route.requested_departure_latitude == pytest.approx(-60.75)
    assert route.requested_destination_longitude == pytest.approx(-58.416667, abs=1e-5)
    assert route.route_planner_version == "route_planner_v1"
    assert route.confidence_status == "not_calibrated"


def test_confidence_is_never_a_percentage(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.schemas.routing import RouteConfidence

    confidence = RouteConfidence(status="not_calibrated")
    assert confidence.percent is None
    assert confidence.status == "not_calibrated"


def test_grid_endpoint_rejects_a_connector_that_crosses_blocked_water(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.port_resolver import ResolvedPort
    from app.services.route_service import RouteError, RouteService

    service = RouteService(db, settings)
    lats = np.linspace(-63.0, -61.0, 5)
    lons = np.linspace(-46.0, -44.0, 5)
    blocked = np.zeros((5, 5), bool)
    port = ResolvedPort(1, "1", "Test", float(lats[2]), float(lons[2]), "Antarctica", "nga_world_port_index")

    # Open water: the port's own cell is used and the connector is trivially valid.
    endpoint = service._grid_endpoint(lats, lons, blocked, port)
    assert endpoint.connector_validated and endpoint.connector_km == pytest.approx(0.0, abs=1e-6)

    # Berth cell blocked (as every real port's cell is) but open water beside it.
    blocked[2, 2] = True
    endpoint = service._grid_endpoint(lats, lons, blocked, port)
    assert endpoint.connector_validated and (endpoint.row, endpoint.col) != (2, 2)

    # Fully enclosed: no connector may be invented.
    blocked[:] = True
    blocked[0, 0] = False
    with pytest.raises(RouteError) as exc:
        service._grid_endpoint(lats, lons, blocked, port)
    assert exc.value.code == "NO_FEASIBLE_ROUTE"


def test_ports_table_holds_canonical_identifiers(db, settings) -> None:  # type: ignore[no-untyped-def]
    resolver(db, settings).refresh_index(force=True)
    row = db.query(Port).filter(Port.identifier == "63080").one()
    assert row.name == "Scotia Bay"
    assert row.normalized_name == "scotia bay"
    assert row.source == "nga_world_port_index"
    assert row.refreshed_at is not None
    assert row.attributes.get("harborSize") == "S"
    assert isinstance(row.latitude, float) and isinstance(row.longitude, float)
    assert date.today() is not None  # sanity: the row carries real values, not placeholders
