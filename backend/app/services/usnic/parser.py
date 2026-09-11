"""Validation and parsing of the USNIC Antarctic iceberg CSV.

Observed format (2026-09-10, https://usicecenter.gov/File/DownloadCurrent?pId=134)::

    Iceberg,Length (NM),Width (NM),Latitude,Longitude,Area (sqMI),Area (sqNM),Area (sqKM),Last Update
    A76C,16,7,-53.94,-26.43,112.44,84.90,291.21,09/10/2026

Columns are matched by *alias*, not position, because the product has changed
over time (earlier editions carried a ``Remarks`` column and no areas).
Coordinates are signed decimal degrees; degree-minute strings with a
hemisphere letter are accepted defensively. ``Last Update`` is US ``MM/DD/YYYY``.

A malformed row is rejected on its own with a reason; it never aborts the file.
A missing required column rejects the whole file (schema error).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta

from ml.adapters.bootstrap_adapter import normalize_iceberg_id

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "iceberg": ("iceberg", "iceberg name", "name", "iceberg id"),
    "length_nm": ("length (nm)", "length(nm)", "length nm", "length"),
    "width_nm": ("width (nm)", "width(nm)", "width nm", "width"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "long"),
    "area_sq_mi": ("area (sqmi)", "area (sq mi)", "area sqmi"),
    "area_sq_nm": ("area (sqnm)", "area (sq nm)", "area sqnm"),
    "area_sq_km": ("area (sqkm)", "area (sq km)", "area sqkm"),
    "last_update": ("last update", "last updated", "lastupdate", "last_update", "date"),
    "remarks": ("remarks", "remark", "comments"),
}
REQUIRED_COLUMNS = ("iceberg", "latitude", "longitude", "last_update")
DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y")
ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,11}$")
_DM_PATTERN = re.compile(
    r"^\s*(?P<deg>\d{1,3}(?:\.\d+)?)\s*(?:°|\s|-|d)?\s*(?:(?P<min>\d{1,2}(?:\.\d+)?)\s*['′m]?)?\s*(?P<hem>[NSEWnsew])\s*$"
)
LATITUDE_RANGE = (-90.0, -30.0)  # Antarctic product: southern hemisphere only
SQ_NM_TO_SQ_KM = 3.429904


class SchemaError(ValueError):
    """The file is not a USNIC Antarctic iceberg CSV we can interpret."""


@dataclass(frozen=True)
class ParsedObservation:
    row_number: int
    iceberg_id: str
    raw_iceberg: str
    observation_date: date
    latitude: float
    longitude: float
    length_nm: float | None
    width_nm: float | None
    area_sq_nm: float | None
    area_sq_km: float | None
    area_sq_mi: float | None
    remarks: str | None

    @property
    def content_hash(self) -> str:
        """Hash of the observed values (not row number) for change detection."""
        payload = {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "length_nm": self.length_nm,
            "width_nm": self.width_nm,
            "area_sq_nm": self.area_sq_nm,
            "area_sq_km": self.area_sq_km,
            "area_sq_mi": self.area_sq_mi,
            "remarks": self.remarks,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RowError:
    row_number: int
    raw_row: dict[str, str]
    errors: tuple[str, ...]


@dataclass
class ParseResult:
    header: list[str]
    column_mapping: dict[str, str]
    observations: list[ParsedObservation] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.observations) + len(self.errors)

    @property
    def latest_update(self) -> date | None:
        return max((o.observation_date for o in self.observations), default=None)

    def summary(self) -> dict[str, object]:
        return {
            "header": self.header,
            "rows": self.row_count,
            "valid": len(self.observations),
            "failed": len(self.errors),
            "latest_update": self.latest_update.isoformat() if self.latest_update else None,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------- helpers
def _normalize_header(h: str) -> str:
    return re.sub(r"\s+", " ", h.replace("﻿", "").strip().lower())


def map_columns(header: list[str]) -> dict[str, str]:
    """canonical name -> header as written in the file."""
    normalized = {_normalize_header(h): h for h in header}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[canonical] = normalized[alias]
                break
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping]
    if missing:
        raise SchemaError(f"missing required column(s) {missing}; header was {header}")
    return mapping


def _decimal_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _degree_minute(text: str, raw: str, axis: str) -> float:
    m = _DM_PATTERN.match(text)
    if not m:
        raise ValueError(f"{axis} {raw!r} is not a recognised coordinate")
    minutes = float(m.group("min") or 0)
    if minutes >= 60:
        raise ValueError(f"{axis} {raw!r} has minutes >= 60")
    hem = m.group("hem").upper()
    if (axis == "latitude" and hem not in "NS") or (axis == "longitude" and hem not in "EW"):
        raise ValueError(f"{axis} {raw!r} has hemisphere {hem}")
    value = float(m.group("deg")) + minutes / 60.0
    return -value if hem in "SW" else value


def parse_coordinate(raw: str, axis: str) -> float:
    """Decimal degrees or degree[-minute] with hemisphere letter."""
    text = (raw or "").strip()
    if not text:
        raise ValueError(f"{axis} is empty")
    value = _decimal_or_none(text)
    if value is None:
        value = _degree_minute(text, raw, axis)
    if axis == "latitude":
        if not LATITUDE_RANGE[0] <= value <= LATITUDE_RANGE[1]:
            raise ValueError(f"latitude {value} outside Antarctic range {LATITUDE_RANGE}")
    else:
        if not -180.0 <= value <= 180.0:
            raise ValueError(f"longitude {value} outside [-180, 180]")
    return value


def parse_usnic_date(raw: str, not_after: date | None = None) -> date:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Last Update is empty")
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Last Update {raw!r} is not MM/DD/YYYY")
    if parsed.year < 1970:
        raise ValueError(f"Last Update {parsed} is implausibly old")
    if not_after is not None and parsed > not_after:
        raise ValueError(f"Last Update {parsed} is in the future (fetched {not_after - timedelta(days=1)})")
    return parsed


def _optional_number(raw: str | None, name: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        raise ValueError(f"{name} {raw!r} is not numeric") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} is not finite")
    if value < 0:
        raise ValueError(f"{name} {value} is negative")
    return value


def decode_csv_bytes(content: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise SchemaError("CSV is not decodable text")


def looks_like_html(text: str) -> bool:
    head = text.lstrip()[:200].lower()
    return head.startswith("<!doctype") or head.startswith("<html") or "<head" in head


# ----------------------------------------------------------------------------- parser
def parse_usnic_csv(content: bytes | str, fetched_on: date | None = None) -> ParseResult:
    text = decode_csv_bytes(content) if isinstance(content, bytes) else content
    if not text.strip():
        raise SchemaError("CSV is empty")
    if looks_like_html(text):
        raise SchemaError("response is an HTML page, not a CSV")

    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise SchemaError("CSV has no header") from None
    header = [h.strip() for h in header]
    mapping = map_columns(header)
    index = {name: header.index(col) for name, col in mapping.items()}
    not_after = (fetched_on + timedelta(days=1)) if fetched_on else None

    result = ParseResult(header=header, column_mapping=mapping)
    seen: dict[tuple[str, date], ParsedObservation] = {}

    for row_number, row in enumerate(reader, start=2):
        if not row or all(not c.strip() for c in row):
            continue
        raw = {header[i] if i < len(header) else f"extra_{i}": v for i, v in enumerate(row)}

        def cell(name: str, row: list[str] = row) -> str | None:
            i = index.get(name)
            return row[i] if i is not None and i < len(row) else None

        errors: list[str] = []
        if len(row) != len(header):
            errors.append(f"expected {len(header)} fields, got {len(row)}")
        raw_id = (cell("iceberg") or "").strip()
        iceberg_id = normalize_iceberg_id(raw_id)
        if not ID_PATTERN.match(iceberg_id):
            errors.append(f"iceberg id {raw_id!r} is not a valid designator")
        values: dict[str, object] = {}
        for axis in ("latitude", "longitude"):
            try:
                values[axis] = parse_coordinate(cell(axis) or "", axis)
            except ValueError as exc:
                errors.append(str(exc))
        try:
            values["observation_date"] = parse_usnic_date(cell("last_update") or "", not_after)
        except ValueError as exc:
            errors.append(str(exc))
        for name in ("length_nm", "width_nm", "area_sq_nm", "area_sq_km", "area_sq_mi"):
            try:
                values[name] = _optional_number(cell(name), name)
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            result.errors.append(RowError(row_number, raw, tuple(errors)))
            continue

        remarks = (cell("remarks") or "").strip() or None
        obs = ParsedObservation(
            row_number=row_number,
            iceberg_id=iceberg_id,
            raw_iceberg=raw_id,
            remarks=remarks,
            **values,  # type: ignore[arg-type]
        )
        key = (obs.iceberg_id, obs.observation_date)
        if key in seen:
            if seen[key].content_hash == obs.content_hash:
                result.warnings.append(f"row {row_number}: exact duplicate of row {seen[key].row_number} ignored")
            else:
                result.errors.append(
                    RowError(row_number, raw, (f"conflicts with row {seen[key].row_number} for {key[0]} on {key[1]}",))
                )
            continue
        if obs.area_sq_nm and obs.area_sq_km and abs(obs.area_sq_nm * SQ_NM_TO_SQ_KM - obs.area_sq_km) > max(1.0, 0.02 * obs.area_sq_km):
            result.warnings.append(f"row {row_number}: area sqNM/sqKM inconsistent for {obs.iceberg_id}")
        seen[key] = obs
        result.observations.append(obs)
    return result


def row_error_payload(err: RowError) -> dict[str, object]:
    return asdict(err)
