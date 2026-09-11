"""Conversion of stored observation rows into adapter entries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from ml.adapters.sequence_builder import TrajectoryEntry
from ml.provenance import Provenance


def entry_from_row(row: Mapping[str, Any]) -> TrajectoryEntry:
    """Build an entry from a mapping with observation columns.

    Required keys: iceberg_id, observation_date, latitude, longitude, provenance.
    Optional: id / observation_id, source.
    """
    obs_date = row["observation_date"]
    if not isinstance(obs_date, date):
        raise TypeError(f"observation_date must be a date, got {type(obs_date).__name__}")
    return TrajectoryEntry(
        iceberg_id=str(row["iceberg_id"]),
        observation_date=obs_date,
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        provenance=Provenance(row["provenance"]),
        observation_id=row.get("observation_id", row.get("id")),
        source=row.get("source"),
    )


def group_entries(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[TrajectoryEntry]]:
    grouped: dict[str, list[TrajectoryEntry]] = defaultdict(list)
    for row in rows:
        entry = entry_from_row(row)
        grouped[entry.iceberg_id].append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda e: e.observation_date)
    return dict(grouped)
