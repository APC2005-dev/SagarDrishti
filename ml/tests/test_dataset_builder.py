from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ml.adapters.production_adapter import ProductionSequenceAdapter
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.provenance import Provenance
from ml.tests.conftest import make_track
from ml.training.dataset_builder import (
    SAMPLE_OPERATIONAL,
    build_historical_sequences,
    build_operational_samples,
    split_by_boundaries,
    split_by_fraction,
)


def _tracks(iceberg: str, start: str, days: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    lat = -65 + np.arange(days) * 0.01
    lon = 40 + np.arange(days) * 0.03
    x, y = latlon_to_polar_m(lat, lon)
    return pd.DataFrame({"iceberg_id": iceberg, "date": dates, "latitude": lat, "longitude": lon, "x_m": x, "y_m": y})


def test_historical_window_count_and_shapes() -> None:
    ds = build_historical_sequences(_tracks("A1", "2020-01-01", 25))
    # windows need 14 history + 7 future: 25 - 21 + 1 = 5
    assert len(ds) == 5
    assert ds.X.shape == (5, 14, 6) and ds.y.shape == (5, 14)
    assert not np.isnan(ds.y).any()
    # target of first window, day 1 = displacement from day 13 to day 14
    t = _tracks("A1", "2020-01-01", 25)
    np.testing.assert_allclose(ds.y[0, 0], (t.x_m[14] - t.x_m[13]) / 1000, rtol=1e-5)


def test_historical_sequences_break_on_gaps() -> None:
    a = _tracks("A1", "2020-01-01", 20)
    b = _tracks("A1", "2020-02-01", 22)  # separate daily run
    ds = build_historical_sequences(pd.concat([a, b]))
    assert len(ds) == 22 - 21 + 1  # the 20-day run is too short


def test_fraction_split_is_chronological_and_date_grouped() -> None:
    frames = [_tracks(f"B{i}", "2020-01-01", 60) for i in range(3)]
    ds = build_historical_sequences(pd.concat(frames))
    split = split_by_fraction(ds, 0.75, 0.125)
    assert len(split.train) + len(split.validation) + len(split.test) == len(ds)
    assert split.train.anchor_dates.max() < split.validation.anchor_dates.min()
    assert split.validation.anchor_dates.max() < split.test.anchor_dates.min()
    for part in (split.train, split.validation, split.test):
        # all 3 icebergs share each date, so each date's samples stay together
        assert len(part) % 3 == 0


def test_boundary_split_validation() -> None:
    ds = build_historical_sequences(_tracks("A1", "2020-01-01", 40))
    with pytest.raises(ValueError):
        split_by_boundaries(ds, np.datetime64("2020-02-01"), np.datetime64("2020-01-01"))


def test_operational_samples_only_label_matched_horizons() -> None:
    hist = make_track("A81", date(2026, 3, 1), 30, 1, Provenance.HISTORICAL_TRAINING_DATASET)
    official = make_track("A81", date(2026, 6, 4), 4, 7, Provenance.OFFICIAL_USNIC, first_id=100)
    ds = build_operational_samples({"A81": hist + official}, ProductionSequenceAdapter())
    # 4 weekly official observations -> 3 anchors have a D+7 official successor
    assert len(ds) == 3
    y = ds.y.reshape(-1, 7, 2)
    assert np.isfinite(y[:, 6]).all()
    assert np.isnan(y[:, :6]).all()
    assert set(ds.sample_kind) == {SAMPLE_OPERATIONAL}
    assert list(ds.anchor_observation_ids) == [100, 101, 102]


def test_operational_samples_respect_cutoff() -> None:
    hist = make_track("A81", date(2026, 3, 1), 30, 1, Provenance.HISTORICAL_TRAINING_DATASET)
    official = make_track("A81", date(2026, 6, 4), 4, 7, Provenance.OFFICIAL_USNIC, first_id=100)
    ds = build_operational_samples(
        {"A81": hist + official}, ProductionSequenceAdapter(), cutoff=date(2026, 6, 4) + timedelta(days=7)
    )
    assert len(ds) == 1


def test_predictions_never_become_labels() -> None:
    hist = make_track("A81", date(2026, 3, 1), 30, 1, Provenance.HISTORICAL_TRAINING_DATASET)
    official = make_track("A81", date(2026, 6, 4), 1, 7, Provenance.OFFICIAL_USNIC, first_id=100)
    predicted = make_track("A81", date(2026, 6, 11), 1, 7, Provenance.PREDICTED, first_id=200)
    ds = build_operational_samples({"A81": hist + official + predicted}, ProductionSequenceAdapter())
    assert len(ds) == 0
