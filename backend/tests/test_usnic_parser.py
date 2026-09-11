from __future__ import annotations

from datetime import date

import pytest

from app.services.usnic.parser import (
    SchemaError,
    map_columns,
    parse_coordinate,
    parse_usnic_csv,
    parse_usnic_date,
)

HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Area (sqMI),Area (sqNM),Area (sqKM),Last Update"


def test_real_usnic_file(fixture_csv: bytes) -> None:
    result = parse_usnic_csv(fixture_csv, fetched_on=date(2026, 9, 11))
    assert result.row_count == 33
    assert len(result.observations) == 33 and not result.errors
    first = result.observations[0]
    assert (first.iceberg_id, first.latitude, first.longitude) == ("A76C", -53.94, -26.43)
    assert first.observation_date == date(2026, 9, 10)
    assert first.length_nm == 16 and first.area_sq_km == pytest.approx(291.21)
    assert result.latest_update == date(2026, 9, 10)
    assert {o.iceberg_id for o in result.observations} >= {"A81", "D15A", "B22F", "D37"}
    b22f = next(o for o in result.observations if o.iceberg_id == "B22F")
    assert b22f.longitude == -176.55


class TestDates:
    def test_us_format(self) -> None:
        assert parse_usnic_date("09/10/2026") == date(2026, 9, 10)
        assert parse_usnic_date("9/4/2026") == date(2026, 9, 4)

    def test_iso_accepted(self) -> None:
        assert parse_usnic_date("2026-09-04") == date(2026, 9, 4)

    @pytest.mark.parametrize("bad", ["", "13/45/2026", "2026/31/12", "yesterday", "02/30/2026"])
    def test_invalid(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_usnic_date(bad)

    def test_future_rejected(self) -> None:
        with pytest.raises(ValueError, match="future"):
            parse_usnic_date("09/20/2026", not_after=date(2026, 9, 12))

    def test_implausibly_old(self) -> None:
        with pytest.raises(ValueError):
            parse_usnic_date("01/01/1901")


class TestCoordinates:
    def test_decimal(self) -> None:
        assert parse_coordinate("-66.07", "latitude") == -66.07
        assert parse_coordinate("143.15", "longitude") == 143.15

    def test_degree_minute_with_hemisphere(self) -> None:
        assert parse_coordinate("66 30S", "latitude") == pytest.approx(-66.5)
        assert parse_coordinate("66°30'S", "latitude") == pytest.approx(-66.5)
        assert parse_coordinate("45 15W", "longitude") == pytest.approx(-45.25)
        assert parse_coordinate("45.5E", "longitude") == pytest.approx(45.5)

    @pytest.mark.parametrize(
        "raw,axis",
        [("", "latitude"), ("abc", "latitude"), ("45.0", "latitude"), ("66 30N", "latitude"), ("-95", "latitude"),
         ("200", "longitude"), ("66 30S", "longitude"), ("66 75S", "latitude")],
    )
    def test_invalid(self, raw: str, axis: str) -> None:
        with pytest.raises(ValueError):
            parse_coordinate(raw, axis)


class TestSchema:
    def test_aliases_and_remarks_column(self) -> None:
        csv = "ICEBERG,Length,Width,Lat,Lon,Remarks,Last Updated\nA-23A,39,34,-60.1,-45.2,grounded?,08/28/2026\n"
        result = parse_usnic_csv(csv)
        [o] = result.observations
        assert o.iceberg_id == "A23A" and o.raw_iceberg == "A-23A"
        assert o.remarks == "grounded?" and o.area_sq_km is None

    def test_bom_header(self) -> None:
        result = parse_usnic_csv(("﻿" + HEADER + "\nA81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026\n").encode("utf-8"))
        assert len(result.observations) == 1

    def test_missing_required_column(self) -> None:
        with pytest.raises(SchemaError, match="latitude"):
            parse_usnic_csv("Iceberg,Longitude,Last Update\nA81,-47.2,09/10/2026\n")

    def test_html_rejected(self) -> None:
        with pytest.raises(SchemaError, match="HTML"):
            parse_usnic_csv("<!DOCTYPE html><html><body>Maintenance</body></html>")

    @pytest.mark.parametrize("content", ["", "   \n"])
    def test_empty(self, content: str) -> None:
        with pytest.raises(SchemaError):
            parse_usnic_csv(content)

    def test_map_columns(self) -> None:
        assert map_columns(HEADER.split(","))["last_update"] == "Last Update"


def test_malformed_rows_are_rejected_individually() -> None:
    rows = [
        "A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026",  # good
        "A83,12,7,abc,-50.61,73.29,55.34,189.82,09/10/2026",  # bad latitude
        "A84,12,6,45.0,-101.65,69.88,52.77,181.00,09/10/2026",  # northern hemisphere
        "A85,10,3,-61.72,200,24.18,18.26,62.62,09/10/2026",  # bad longitude
        "B09B,27,10,-66.07,143.15,196.50,148.38,508.94,13/45/2026",  # bad date
        "B09G,-12,7,-68.18,41.5,61.80,46.66,160.06,09/10/2026",  # negative length
        "B22A,29,25,-69.88,164.79,549.47",  # short row
        "12,8,6,-70.21,163.7,29.14,22.00,75.47,09/10/2026",  # invalid designator
        "B51,15,3,-74.25,-131.68,30.50,23.03,79.00,09/30/2026",  # future
        "C15,x,10,-65.84,143.02,82.44,62.25,213.53,09/10/2026",  # non-numeric length
        "",  # blank line ignored
    ]
    result = parse_usnic_csv(HEADER + "\n" + "\n".join(rows) + "\n", fetched_on=date(2026, 9, 11))
    assert [o.iceberg_id for o in result.observations] == ["A81"]
    assert len(result.errors) == 9
    assert result.row_count == 10
    assert all(e.errors for e in result.errors)
    assert result.errors[0].raw_row["Iceberg"] == "A83"


def test_duplicate_rows() -> None:
    same = "A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026"
    conflict = "A81,28,25,-57.50,-47.22,518.07,391.20,1341.79,09/10/2026"
    result = parse_usnic_csv("\n".join([HEADER, same, same, conflict]) + "\n")
    assert len(result.observations) == 1
    assert len(result.warnings) == 1 and "duplicate" in result.warnings[0]
    assert len(result.errors) == 1 and "conflicts" in result.errors[0].errors[0]


def test_content_hash_ignores_row_position() -> None:
    a = parse_usnic_csv(HEADER + "\nA81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026\n").observations[0]
    b = parse_usnic_csv(HEADER + "\nA83,12,7,-61.07,-50.61,73.29,55.34,189.82,09/10/2026\nA81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026\n").observations[1]
    assert a.content_hash == b.content_hash
    c = parse_usnic_csv(HEADER + "\nA81,28,25,-57.37,-47.22,518.07,391.20,1341.79,09/10/2026\n").observations[0]
    assert a.content_hash != c.content_hash
