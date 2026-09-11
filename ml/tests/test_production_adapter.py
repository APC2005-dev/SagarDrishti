from __future__ import annotations

from datetime import date

import pytest

from ml.adapters.production_adapter import (
    STRATEGY_OFFICIAL_ONLY,
    AdapterConfig,
    ProductionSequenceAdapter,
)
from ml.adapters.sequence_builder import ModelInputSequence, SequenceSkip, SkipReason, TrajectoryEntry
from ml.provenance import Provenance
from ml.tests.conftest import make_track

HIST = Provenance.HISTORICAL_TRAINING_DATASET
OFF = Provenance.OFFICIAL_USNIC


def test_bootstrap_uses_latest_official_plus_13_historical() -> None:
    hist = make_track("A81", date(2026, 4, 1), 30, 1, HIST)
    official = make_track("A81", date(2026, 9, 10), 1, 7, OFF, lat0=-57.36, lon0=-47.22, first_id=1000)
    seq = ProductionSequenceAdapter().build(hist + official)
    assert isinstance(seq, ModelInputSequence)
    assert len(seq.entries) == 14
    assert seq.anchor.provenance == OFF and seq.anchor.observation_date == date(2026, 9, 10)
    assert [e.provenance for e in seq.entries[:13]] == [HIST] * 13
    # historical entries are the most recent 13 before the official one, chronological
    assert seq.entries[0].observation_date == date(2026, 4, 18)
    dates = [e.observation_date for e in seq.entries]
    assert dates == sorted(dates)
    assert seq.diagnostics.official_entries == 1
    assert seq.diagnostics.historical_entries == 13
    assert not seq.diagnostics.daily_cadence
    assert seq.diagnostics.max_gap_days == (date(2026, 9, 10) - date(2026, 4, 30)).days


def test_official_entries_push_out_historical() -> None:
    hist = make_track("A81", date(2026, 1, 1), 40, 1, HIST)
    official = make_track("A81", date(2026, 6, 4), 5, 7, OFF, first_id=500)
    seq = ProductionSequenceAdapter().build(hist + official)
    assert isinstance(seq, ModelInputSequence)
    assert seq.diagnostics.official_entries == 5
    assert seq.diagnostics.historical_entries == 9
    assert all(e.provenance == OFF for e in seq.entries[-5:])


def test_fourteen_official_entries_need_no_bootstrap() -> None:
    hist = make_track("A81", date(2025, 1, 1), 40, 1, HIST)
    official = make_track("A81", date(2026, 1, 1), 20, 7, OFF, first_id=500)
    seq = ProductionSequenceAdapter().build(hist + official)
    assert isinstance(seq, ModelInputSequence)
    assert seq.diagnostics.official_entries == 14
    assert seq.entries[-1].observation_date == official[-1].observation_date


def test_predicted_and_interpolated_are_never_inputs() -> None:
    hist = make_track("A81", date(2026, 1, 1), 5, 1, HIST)
    predicted = make_track("A81", date(2026, 2, 1), 20, 1, Provenance.PREDICTED)
    interp = make_track("A81", date(2026, 3, 1), 20, 1, Provenance.INTERPOLATED)
    official = make_track("A81", date(2026, 9, 1), 1, 7, OFF, first_id=900)
    result = ProductionSequenceAdapter().build(hist + predicted + interp + official)
    assert isinstance(result, SequenceSkip)
    assert result.reason == SkipReason.INSUFFICIENT_HISTORY
    assert result.available_entries == 6


def test_official_wins_on_same_date() -> None:
    hist = make_track("A81", date(2026, 1, 1), 20, 1, HIST)
    dup = TrajectoryEntry("A81", hist[-1].observation_date, -60.0, 10.0, OFF, observation_id=77)
    seq = ProductionSequenceAdapter().build([*hist, dup])
    assert isinstance(seq, ModelInputSequence)
    assert seq.anchor.observation_id == 77
    assert sum(e.observation_date == dup.observation_date for e in seq.entries) == 1


def test_no_official_observation_is_skipped() -> None:
    result = ProductionSequenceAdapter().build(make_track("X1", date(2026, 1, 1), 30, 1, HIST))
    assert isinstance(result, SequenceSkip) and result.reason == SkipReason.NO_OFFICIAL_OBSERVATION


def test_official_only_strategy_requires_14_official() -> None:
    adapter = ProductionSequenceAdapter(AdapterConfig(strategy=STRATEGY_OFFICIAL_ONLY))
    hist = make_track("A81", date(2026, 1, 1), 30, 1, HIST)
    official = make_track("A81", date(2026, 6, 1), 3, 7, OFF, first_id=500)
    result = adapter.build(hist + official)
    assert isinstance(result, SequenceSkip) and result.reason == SkipReason.INSUFFICIENT_HISTORY


def test_gap_too_large() -> None:
    hist = make_track("A81", date(2015, 1, 1), 20, 1, HIST)
    official = make_track("A81", date(2026, 9, 10), 1, 7, OFF, first_id=500)
    result = ProductionSequenceAdapter(AdapterConfig(max_gap_days=730)).build(hist + official)
    assert isinstance(result, SequenceSkip) and result.reason == SkipReason.GAP_TOO_LARGE


def test_explicit_anchor_ignores_later_data() -> None:
    hist = make_track("A81", date(2026, 1, 1), 30, 1, HIST)
    official = make_track("A81", date(2026, 6, 1), 4, 7, OFF, first_id=500)
    seq = ProductionSequenceAdapter().build(hist + official, anchor=official[1])
    assert isinstance(seq, ModelInputSequence)
    assert seq.anchor.observation_id == official[1].observation_id
    assert all(e.observation_date <= official[1].observation_date for e in seq.entries)


def test_anchor_must_be_official() -> None:
    hist = make_track("A81", date(2026, 1, 1), 30, 1, HIST)
    with pytest.raises(ValueError):
        ProductionSequenceAdapter().build(hist, anchor=hist[-1])


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValueError):
        AdapterConfig(strategy="made_up")


def test_build_many_collects_skips() -> None:
    good = make_track("A", date(2026, 1, 1), 20, 1, HIST) + make_track("A", date(2026, 3, 1), 1, 7, OFF, first_id=99)
    bad = make_track("B", date(2026, 1, 1), 3, 1, OFF)
    result = ProductionSequenceAdapter().build_many({"A": good, "B": bad})
    assert [s.iceberg_id for s in result.sequences] == ["A"]
    assert [s.iceberg_id for s in result.skipped] == ["B"]
