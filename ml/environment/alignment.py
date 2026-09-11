"""Attach environmental columns to sequence datasets (training / evaluation).

Every sample is aligned on its own entries' positions and dates, with the
prediction time ``T`` = the sample's anchor date (see the as-of rule in
``feature_builder``). Output columns are ``ALL_FEATURES`` (trajectory first),
so each model selects its own schema's columns from the same samples — the
basis for comparing candidates on identical data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.environment.feature_builder import EnvironmentalFeatureBuilder
from ml.features.schemas import ALL_ENV_FEATURES, column_indices
from ml.training.dataset_builder import SequenceDataset


@dataclass
class AlignmentReport:
    samples: int
    variables: tuple[str, ...]
    missing_entries_by_variable: dict[str, int] = field(default_factory=dict)
    incomplete_samples_by_variable: dict[str, int] = field(default_factory=dict)
    providers: dict[str, str] = field(default_factory=dict)

    def missing_rate(self) -> dict[str, float]:
        total = max(self.samples * 14, 1)
        return {v: n / total for v, n in self.missing_entries_by_variable.items()}

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "variables": list(self.variables),
            "missing_entries_by_variable": self.missing_entries_by_variable,
            "missing_rate_by_variable": self.missing_rate(),
            "incomplete_samples_by_variable": self.incomplete_samples_by_variable,
            "providers": self.providers,
        }


def attach_environment(
    ds: SequenceDataset,
    builder: EnvironmentalFeatureBuilder,
    variables: Sequence[str] = ALL_ENV_FEATURES,
) -> tuple[SequenceDataset, AlignmentReport]:
    if ds.entry_lat is None or ds.entry_lon is None or ds.entry_date is None:
        raise ValueError("dataset lacks per-entry coordinates (build it with with_entries=True)")
    variables = tuple(variables)
    n, steps = ds.X.shape[:2]
    env = np.full((n, steps, len(variables)), np.nan, dtype=np.float32)
    for k in range(n):
        as_of = pd.Timestamp(ds.anchor_dates[k]).date()
        entries = [
            (float(ds.entry_lat[k, i]), float(ds.entry_lon[k, i]), pd.Timestamp(ds.entry_date[k, i]).date())
            for i in range(steps)
        ]
        env[k] = builder.build(entries, as_of, variables).values
    X_full = np.concatenate([ds.X, env], axis=-1).astype(np.float32)
    names = tuple(ds.feature_names) + variables
    out = ds.with_features(X_full, names)
    report = AlignmentReport(
        samples=n,
        variables=variables,
        missing_entries_by_variable={v: int((~np.isfinite(env[..., j])).sum()) for j, v in enumerate(variables)},
        incomplete_samples_by_variable={v: int((~np.isfinite(env[..., j])).any(axis=1).sum()) for j, v in enumerate(variables)},
        providers={g.value: p.spec.name for g, p in builder.providers.items()},
    )
    return out, report


def complete_mask(ds: SequenceDataset, features: Sequence[str]) -> np.ndarray:
    """Samples whose listed features are finite at every entry (complete-case rule)."""
    idx = column_indices(ds.feature_names, features)
    return np.isfinite(ds.X[..., idx]).all(axis=(1, 2))
