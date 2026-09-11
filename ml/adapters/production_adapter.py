"""ProductionSequenceAdapter — operational observations -> exact GRU tensor.

WHY THIS EXISTS
---------------
The base GRU learned from *14 consecutive daily positions* (BYU consolidated
database, notebook §3). The operational USNIC Antarctic iceberg product is
*weekly*. The GRU's sequence axis is kept at 14, but in production it means
"14 chronological observation entries", not "14 calendar days". This adapter is
the single place where that semantic change happens, and it records exactly
what it did so the difference is never hidden downstream.

SEQUENCE SEMANTICS
------------------
* 14 entries, strictly increasing observation dates, oldest first.
* The last entry (tensor index 13, "entry 1" counting back from now) is the
  **anchor**: always the latest official USNIC observation of that iceberg.
  Forecast day D+h means anchor_date + h calendar days.
* Only provenance ``official_usnic`` and ``historical_training_dataset`` may
  appear. Predicted / interpolated / derived positions are never inputs.
* If several entries share a date, the official one wins.

TIME INTERVALS AND FEATURES
---------------------------
Gaps between entries are whatever the sources provide (1 day inside BYU daily
runs, ~7 days between USNIC updates, possibly months at the BYU -> USNIC seam).
Features are built by :func:`ml.features.trajectory_features.build_feature_matrix`:
velocity = displacement / *actual elapsed days*, never displacement / entry.
Relative positions and the seasonal cycle are computed per entry exactly as in
the notebook. Diagnostics (gap statistics, entry provenance counts, whether the
cadence is daily) are attached to every sequence and persisted with forecasts.

STRATEGIES / BOOTSTRAP
----------------------
``bootstrap_v1``  (model v1 at first deployment)
    Take the most recent official USNIC observations (at least the anchor). If
    fewer than 14 exist, fill the *earlier* slots with the iceberg's most recent
    historical-training-dataset positions dated strictly before the earliest
    official entry used. On day one that is: anchor = latest USNIC
    observation, entries 1-13 = BYU history. As weekly USNIC observations
    accumulate, historical entries are pushed out automatically, so the input
    distribution drifts towards all-official without any code change.

``official_only``
    Require 14 official USNIC observations. Intended for future model versions
    trained predominantly on operational data.

No gap is filled by interpolation. If an iceberg lacks enough history, it is
skipped with an explicit reason rather than forecast from invented positions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ml.adapters.sequence_builder import (
    AdapterResult,
    ModelInputSequence,
    SequenceDiagnostics,
    SequenceSkip,
    SkipReason,
    TrajectoryEntry,
)
from ml.constants import SEQUENCE_LENGTH
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.features.trajectory_features import FeatureConstructionError, build_feature_matrix
from ml.provenance import MODEL_INPUT_PROVENANCE, Provenance

STRATEGY_BOOTSTRAP_V1 = "bootstrap_v1"
STRATEGY_OFFICIAL_ONLY = "official_only"
ADAPTER_VERSION = "production_sequence_adapter_v1"


@dataclass(frozen=True)
class AdapterConfig:
    strategy: str = STRATEGY_BOOTSTRAP_V1
    # Reject sequences containing a gap larger than this (days). A multi-year
    # hole means the "history" no longer describes the current drift regime.
    max_gap_days: float = 730.0

    def __post_init__(self) -> None:
        if self.strategy not in (STRATEGY_BOOTSTRAP_V1, STRATEGY_OFFICIAL_ONLY):
            raise ValueError(f"unknown adapter strategy {self.strategy!r}")


class ProductionSequenceAdapter:
    def __init__(self, config: AdapterConfig | None = None) -> None:
        self.config = config or AdapterConfig()

    # ------------------------------------------------------------------ public
    def build(self, entries: Iterable[TrajectoryEntry], anchor: TrajectoryEntry | None = None) -> ModelInputSequence | SequenceSkip:
        """Build the sequence for one iceberg.

        ``anchor`` pins the sequence end to a specific official observation
        (used when building retraining samples at historical anchors). By
        default the latest official observation is used.
        """
        entries = list(entries)
        iceberg_id = entries[0].iceberg_id if entries else (anchor.iceberg_id if anchor else "?")
        usable = self._dedupe([e for e in entries if e.provenance in MODEL_INPUT_PROVENANCE])
        official = [e for e in usable if e.provenance == Provenance.OFFICIAL_USNIC]

        if anchor is None:
            if not official:
                return SequenceSkip(iceberg_id, SkipReason.NO_OFFICIAL_OBSERVATION, available_entries=len(usable))
            anchor = official[-1]
        elif anchor.provenance != Provenance.OFFICIAL_USNIC:
            raise ValueError("anchor must be an official_usnic observation")

        official_upto = [e for e in official if e.observation_date <= anchor.observation_date]
        chosen_official = official_upto[-SEQUENCE_LENGTH:]
        if not chosen_official or chosen_official[-1].observation_date != anchor.observation_date:
            chosen_official = [*chosen_official, anchor][-SEQUENCE_LENGTH:]

        chosen: list[TrajectoryEntry] = list(chosen_official)
        if len(chosen) < SEQUENCE_LENGTH:
            if self.config.strategy == STRATEGY_OFFICIAL_ONLY:
                return SequenceSkip(
                    iceberg_id,
                    SkipReason.INSUFFICIENT_HISTORY,
                    f"official_only needs {SEQUENCE_LENGTH} official entries, have {len(chosen)}",
                    len(chosen),
                )
            earliest = chosen[0].observation_date
            historical = [
                e
                for e in usable
                if e.provenance == Provenance.HISTORICAL_TRAINING_DATASET and e.observation_date < earliest
            ]
            need = SEQUENCE_LENGTH - len(chosen)
            chosen = historical[-need:] + chosen

        if len(chosen) < SEQUENCE_LENGTH:
            return SequenceSkip(
                iceberg_id,
                SkipReason.INSUFFICIENT_HISTORY,
                f"need {SEQUENCE_LENGTH} entries, have {len(chosen)}",
                len(chosen),
            )

        lats = np.asarray([e.latitude for e in chosen], dtype=np.float64)
        lons = np.asarray([e.longitude for e in chosen], dtype=np.float64)
        if not (np.all(np.abs(lats) <= 90) and np.all(np.abs(lons) <= 180)):
            return SequenceSkip(iceberg_id, SkipReason.INVALID_COORDINATES, available_entries=len(chosen))
        x_m, y_m = latlon_to_polar_m(lats, lons)
        dates = [e.observation_date for e in chosen]
        try:
            fm = build_feature_matrix(x_m, y_m, dates)
        except FeatureConstructionError as exc:
            return SequenceSkip(iceberg_id, SkipReason.INVALID_COORDINATES, str(exc), len(chosen))

        gaps = fm.elapsed_days[1:]
        if gaps.max() > self.config.max_gap_days:
            return SequenceSkip(
                iceberg_id,
                SkipReason.GAP_TOO_LARGE,
                f"max gap {gaps.max():.0f} d > {self.config.max_gap_days:.0f} d",
                len(chosen),
            )

        n_official = sum(e.provenance == Provenance.OFFICIAL_USNIC for e in chosen)
        diagnostics = SequenceDiagnostics(
            official_entries=n_official,
            historical_entries=len(chosen) - n_official,
            min_gap_days=float(gaps.min()),
            median_gap_days=float(np.median(gaps)),
            max_gap_days=float(gaps.max()),
            span_days=float(gaps.sum()),
            daily_cadence=bool(np.all(gaps == 1.0)),
        )
        return ModelInputSequence(
            iceberg_id=iceberg_id,
            entries=tuple(chosen),
            features=fm.features,
            elapsed_days=fm.elapsed_days,
            anchor_x_m=fm.anchor_x_m,
            anchor_y_m=fm.anchor_y_m,
            diagnostics=diagnostics,
            adapter_strategy=self.config.strategy,
        )

    def build_many(self, entries_by_iceberg: dict[str, list[TrajectoryEntry]]) -> AdapterResult:
        result = AdapterResult()
        for iceberg_id, entries in entries_by_iceberg.items():
            built = self.build(entries) if entries else SequenceSkip(iceberg_id, SkipReason.NO_OFFICIAL_OBSERVATION)
            if isinstance(built, ModelInputSequence):
                result.sequences.append(built)
            else:
                result.skipped.append(built)
        return result

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _dedupe(entries: list[TrajectoryEntry]) -> list[TrajectoryEntry]:
        """One entry per date, official preferred, sorted chronologically."""
        by_date: dict[object, TrajectoryEntry] = {}
        for e in entries:
            current = by_date.get(e.observation_date)
            if current is None or (
                current.provenance != Provenance.OFFICIAL_USNIC and e.provenance == Provenance.OFFICIAL_USNIC
            ):
                by_date[e.observation_date] = e
        return [by_date[d] for d in sorted(by_date)]  # type: ignore[type-var]
