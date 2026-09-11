"""Data types shared by the sequence adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from ml.provenance import Provenance


@dataclass(frozen=True)
class TrajectoryEntry:
    """One dated position of one iceberg, with its provenance.

    ``observation_id`` is the database id when the entry came from
    ``tracking.observations``; it is what makes forecast lineage traceable.
    """

    iceberg_id: str
    observation_date: date
    latitude: float
    longitude: float
    provenance: Provenance
    observation_id: int | None = None
    source: str | None = None


class SkipReason(StrEnum):
    NO_OFFICIAL_OBSERVATION = "no_official_observation"
    INSUFFICIENT_HISTORY = "insufficient_history"
    GAP_TOO_LARGE = "gap_too_large"
    INVALID_COORDINATES = "invalid_coordinates"


@dataclass(frozen=True)
class SequenceDiagnostics:
    official_entries: int
    historical_entries: int
    min_gap_days: float
    median_gap_days: float
    max_gap_days: float
    span_days: float
    # True only when every gap is exactly one day, i.e. the cadence the
    # research model was trained on. Weekly USNIC cadence is out of the
    # research training distribution and is surfaced, never hidden.
    daily_cadence: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "official_entries": self.official_entries,
            "historical_entries": self.historical_entries,
            "min_gap_days": self.min_gap_days,
            "median_gap_days": self.median_gap_days,
            "max_gap_days": self.max_gap_days,
            "span_days": self.span_days,
            "daily_cadence": self.daily_cadence,
        }


@dataclass(frozen=True)
class ModelInputSequence:
    """A ready-to-scale GRU input plus everything needed for lineage."""

    iceberg_id: str
    entries: tuple[TrajectoryEntry, ...]  # chronological, len == SEQUENCE_LENGTH
    features: NDArray[np.float32]  # (14, 6), unscaled
    elapsed_days: NDArray[np.float64]
    anchor_x_m: float
    anchor_y_m: float
    diagnostics: SequenceDiagnostics
    adapter_strategy: str

    @property
    def anchor(self) -> TrajectoryEntry:
        return self.entries[-1]


@dataclass(frozen=True)
class SequenceSkip:
    iceberg_id: str
    reason: SkipReason
    detail: str = ""
    available_entries: int = 0


@dataclass
class AdapterResult:
    sequences: list[ModelInputSequence] = field(default_factory=list)
    skipped: list[SequenceSkip] = field(default_factory=list)
