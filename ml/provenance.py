"""Provenance vocabulary for every position the system touches."""

from __future__ import annotations

from enum import StrEnum


class Provenance(StrEnum):
    OFFICIAL_USNIC = "official_usnic"
    HISTORICAL_TRAINING_DATASET = "historical_training_dataset"
    DERIVED = "derived"
    INTERPOLATED = "interpolated"
    PREDICTED = "predicted"


# Positions allowed inside a model input sequence. Predicted, interpolated and
# derived positions are never fed back into the model as if they were observed.
MODEL_INPUT_PROVENANCE: frozenset[Provenance] = frozenset(
    {Provenance.OFFICIAL_USNIC, Provenance.HISTORICAL_TRAINING_DATASET}
)

# The only provenance accepted as ground truth for operational evaluation.
GROUND_TRUTH_PROVENANCE: frozenset[Provenance] = frozenset({Provenance.OFFICIAL_USNIC})
