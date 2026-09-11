from __future__ import annotations

import importlib.util
from datetime import date, timedelta

import pytest

from ml.adapters.sequence_builder import TrajectoryEntry
from ml.provenance import Provenance

HAS_TF = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("keras") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_TF:
        return
    skip = pytest.mark.skip(reason="TensorFlow/Keras not installed")
    for item in items:
        if "tf" in item.keywords:
            item.add_marker(skip)


def make_track(
    iceberg_id: str,
    start: date,
    n: int,
    step_days: int,
    provenance: Provenance,
    lat0: float = -65.0,
    lon0: float = 40.0,
    dlat: float = 0.02,
    dlon: float = 0.05,
    first_id: int = 1,
) -> list[TrajectoryEntry]:
    return [
        TrajectoryEntry(
            iceberg_id=iceberg_id,
            observation_date=start + timedelta(days=i * step_days),
            latitude=lat0 + i * dlat,
            longitude=lon0 + i * dlon,
            provenance=provenance,
            observation_id=first_id + i,
        )
        for i in range(n)
    ]
