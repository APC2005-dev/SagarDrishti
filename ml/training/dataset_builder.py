"""Labelled sequence datasets and chronological splits.

Two sample families share one tensor contract:

* **historical** — notebook §3: every 14-day window inside an
  uninterrupted *daily* run of the historical training dataset, with the next 7
  daily positions as fully-observed targets.
* **operational** — anchored on an official USNIC observation; the 14-entry
  input comes from :class:`ProductionSequenceAdapter` using only data dated on or
  before the anchor, and targets exist only at horizons where a later *official*
  observation falls exactly on anchor + h days (other horizons are NaN). These
  are precisely the (forecast, actual) pairs that the evaluation pipeline
  scores, so evaluated predictions flow into retraining with full lineage.

Splits are always chronological by anchor date — never shuffled — and every
sample from the same anchor date stays in the same split (notebook §4).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml.adapters.production_adapter import ProductionSequenceAdapter
from ml.adapters.sequence_builder import ModelInputSequence, TrajectoryEntry
from ml.constants import FORECAST_DAYS, N_FEATURES, OUTPUT_SIZE, SEQUENCE_LENGTH
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.features.trajectory_features import build_target_vector
from ml.provenance import Provenance

SAMPLE_HISTORICAL = "historical"
SAMPLE_OPERATIONAL = "operational"


@dataclass
class SequenceDataset:
    X: NDArray[np.float32]  # (n, 14, 6) unscaled
    y: NDArray[np.float32]  # (n, 14) km displacement, NaN = unknown horizon
    anchor_dates: NDArray[np.datetime64]
    anchor_x_m: NDArray[np.float64]
    anchor_y_m: NDArray[np.float64]
    iceberg_ids: NDArray[np.object_]
    sample_kind: NDArray[np.object_]
    anchor_observation_ids: NDArray[np.object_] = field(default_factory=lambda: np.zeros(0, dtype=object))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @classmethod
    def empty(cls) -> SequenceDataset:
        return cls(
            X=np.zeros((0, SEQUENCE_LENGTH, N_FEATURES), np.float32),
            y=np.zeros((0, OUTPUT_SIZE), np.float32),
            anchor_dates=np.zeros(0, "datetime64[ns]"),
            anchor_x_m=np.zeros(0),
            anchor_y_m=np.zeros(0),
            iceberg_ids=np.zeros(0, object),
            sample_kind=np.zeros(0, object),
            anchor_observation_ids=np.zeros(0, object),
        )

    def subset(self, mask: NDArray[np.bool_]) -> SequenceDataset:
        ids = self.anchor_observation_ids if len(self.anchor_observation_ids) == len(self) else np.full(len(self), None, object)
        return SequenceDataset(
            X=self.X[mask],
            y=self.y[mask],
            anchor_dates=self.anchor_dates[mask],
            anchor_x_m=self.anchor_x_m[mask],
            anchor_y_m=self.anchor_y_m[mask],
            iceberg_ids=self.iceberg_ids[mask],
            sample_kind=self.sample_kind[mask],
            anchor_observation_ids=ids[mask],
        )

    @staticmethod
    def concat(parts: Iterable[SequenceDataset]) -> SequenceDataset:
        parts = [p for p in parts if len(p)]
        if not parts:
            return SequenceDataset.empty()
        def ids(p: SequenceDataset) -> NDArray[np.object_]:
            return p.anchor_observation_ids if len(p.anchor_observation_ids) == len(p) else np.full(len(p), None, object)
        return SequenceDataset(
            X=np.concatenate([p.X for p in parts]),
            y=np.concatenate([p.y for p in parts]),
            anchor_dates=np.concatenate([p.anchor_dates for p in parts]),
            anchor_x_m=np.concatenate([p.anchor_x_m for p in parts]),
            anchor_y_m=np.concatenate([p.anchor_y_m for p in parts]),
            iceberg_ids=np.concatenate([p.iceberg_ids for p in parts]),
            sample_kind=np.concatenate([p.sample_kind for p in parts]),
            anchor_observation_ids=np.concatenate([ids(p) for p in parts]),
        )

    def date_range(self) -> tuple[str | None, str | None]:
        if not len(self):
            return None, None
        return str(pd.Timestamp(self.anchor_dates.min()).date()), str(pd.Timestamp(self.anchor_dates.max()).date())

    def summary(self) -> dict[str, object]:
        start, end = self.date_range()
        labelled = np.isfinite(self.y.reshape(-1, FORECAST_DAYS, 2)[..., 0]).sum(axis=0) if len(self) else np.zeros(7)
        return {
            "samples": len(self),
            "icebergs": int(len(set(self.iceberg_ids.tolist()))),
            "anchor_date_start": start,
            "anchor_date_end": end,
            "labelled_per_horizon": {str(h + 1): int(labelled[h]) for h in range(FORECAST_DAYS)},
            "kinds": {k: int((self.sample_kind == k).sum()) for k in set(self.sample_kind.tolist())},
        }


# --------------------------------------------------------------------------- historical
def build_historical_sequences(tracks: pd.DataFrame) -> SequenceDataset:
    """Notebook §3: 14-day inputs -> 7-day targets inside daily runs.

    ``tracks`` needs iceberg_id, date (datetime64), x_m, y_m.
    """
    X_list: list[NDArray[np.float32]] = []
    y_list: list[NDArray[np.float32]] = []
    dates, ax, ay, ids = [], [], [], []
    t = tracks.sort_values(["iceberg_id", "date"], kind="mergesort").reset_index(drop=True)
    for iceberg_id, group in t.groupby("iceberg_id", sort=True):
        group = group.reset_index(drop=True)
        run_id = group["date"].diff().dt.days.ne(1).cumsum()
        for _, run in group.groupby(run_id):
            if len(run) < SEQUENCE_LENGTH + FORECAST_DAYS:
                continue
            x = run["x_m"].to_numpy(np.float64)
            y = run["y_m"].to_numpy(np.float64)
            d = run["date"].to_numpy("datetime64[ns]")
            doy = run["date"].dt.dayofyear.to_numpy(np.float64)
            angle = 2 * np.pi * doy / 365.25
            s_sin, s_cos = np.sin(angle), np.cos(angle)
            vx_all = np.diff(x) / 1000.0  # daily run: elapsed days == 1
            vy_all = np.diff(y) / 1000.0
            for end in range(SEQUENCE_LENGTH - 1, len(run) - FORECAST_DAYS):
                start = end - SEQUENCE_LENGTH + 1
                hx, hy = x[start : end + 1], y[start : end + 1]
                cx, cy = hx[-1], hy[-1]
                vx = np.empty(SEQUENCE_LENGTH)
                vy = np.empty(SEQUENCE_LENGTH)
                vx[1:] = vx_all[start:end]
                vy[1:] = vy_all[start:end]
                vx[0], vy[0] = vx[1], vy[1]
                X_list.append(
                    np.column_stack(
                        [(hx - cx) / 1000.0, (hy - cy) / 1000.0, vx, vy, s_sin[start : end + 1], s_cos[start : end + 1]]
                    ).astype(np.float32)
                )
                fx, fy = x[end + 1 : end + 1 + FORECAST_DAYS], y[end + 1 : end + 1 + FORECAST_DAYS]
                y_list.append(np.column_stack([(fx - cx) / 1000.0, (fy - cy) / 1000.0]).reshape(-1).astype(np.float32))
                dates.append(d[end])
                ax.append(cx)
                ay.append(cy)
                ids.append(iceberg_id)
    if not X_list:
        return SequenceDataset.empty()
    n = len(X_list)
    return SequenceDataset(
        X=np.stack(X_list),
        y=np.stack(y_list),
        anchor_dates=np.asarray(dates, dtype="datetime64[ns]"),
        anchor_x_m=np.asarray(ax),
        anchor_y_m=np.asarray(ay),
        iceberg_ids=np.asarray(ids, dtype=object),
        sample_kind=np.full(n, SAMPLE_HISTORICAL, dtype=object),
        anchor_observation_ids=np.full(n, None, dtype=object),
    )


# -------------------------------------------------------------------------- operational
def build_operational_samples(
    entries_by_iceberg: dict[str, list[TrajectoryEntry]],
    adapter: ProductionSequenceAdapter,
    cutoff: date | None = None,
) -> SequenceDataset:
    """Samples anchored on official observations with official targets 1..7 days later.

    ``cutoff`` excludes every observation dated after it (training data cutoff).
    """
    X_list, y_list, dates, ax, ay, ids, obs_ids = [], [], [], [], [], [], []
    for iceberg_id, entries in entries_by_iceberg.items():
        usable = [e for e in entries if cutoff is None or e.observation_date <= cutoff]
        official = sorted(
            (e for e in usable if e.provenance == Provenance.OFFICIAL_USNIC), key=lambda e: e.observation_date
        )
        by_date = {e.observation_date: e for e in official}
        for anchor in official:
            future = [by_date.get(anchor.observation_date + timedelta(days=h)) for h in range(1, FORECAST_DAYS + 1)]
            if not any(future):
                continue
            history = [e for e in usable if e.observation_date <= anchor.observation_date]
            seq = adapter.build(history, anchor=anchor)
            if not isinstance(seq, ModelInputSequence):
                continue
            fx: list[float | None] = []
            fy: list[float | None] = []
            for f in future:
                if f is None:
                    fx.append(None)
                    fy.append(None)
                else:
                    px, py = latlon_to_polar_m([f.latitude], [f.longitude])
                    fx.append(float(px[0]))
                    fy.append(float(py[0]))
            X_list.append(seq.features)
            y_list.append(build_target_vector(seq.anchor_x_m, seq.anchor_y_m, fx, fy))
            dates.append(np.datetime64(anchor.observation_date, "ns"))
            ax.append(seq.anchor_x_m)
            ay.append(seq.anchor_y_m)
            ids.append(iceberg_id)
            obs_ids.append(anchor.observation_id)
    if not X_list:
        return SequenceDataset.empty()
    n = len(X_list)
    return SequenceDataset(
        X=np.stack(X_list).astype(np.float32),
        y=np.stack(y_list).astype(np.float32),
        anchor_dates=np.asarray(dates, dtype="datetime64[ns]"),
        anchor_x_m=np.asarray(ax),
        anchor_y_m=np.asarray(ay),
        iceberg_ids=np.asarray(ids, dtype=object),
        sample_kind=np.full(n, SAMPLE_OPERATIONAL, dtype=object),
        anchor_observation_ids=np.asarray(obs_ids, dtype=object),
    )


# ------------------------------------------------------------------------------ splits
@dataclass
class ChronologicalSplit:
    train: SequenceDataset
    validation: SequenceDataset
    test: SequenceDataset
    train_end: str | None
    validation_end: str | None

    def summary(self) -> dict[str, object]:
        return {
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "train": self.train.summary(),
            "validation": self.validation.summary(),
            "test": self.test.summary(),
        }


def split_by_fraction(ds: SequenceDataset, train_frac: float = 0.75, validation_frac: float = 0.125) -> ChronologicalSplit:
    """Notebook §4: date-grouped cumulative-count boundaries."""
    if not 0 < train_frac < 1 or not 0 < validation_frac < 1 or train_frac + validation_frac >= 1:
        raise ValueError("fractions must satisfy 0 < train, validation and train + validation < 1")
    if len(ds) == 0:
        e = SequenceDataset.empty()
        return ChronologicalSplit(e, e, e, None, None)
    counts = pd.Series(ds.anchor_dates).value_counts().sort_index()
    cumulative = counts.cumsum().to_numpy()
    idx = counts.index
    t_i = min(int(np.searchsorted(cumulative, len(ds) * train_frac)), len(idx) - 1)
    v_i = min(int(np.searchsorted(cumulative, len(ds) * (train_frac + validation_frac))), len(idx) - 1)
    return split_by_boundaries(ds, np.datetime64(idx[t_i], "ns"), np.datetime64(idx[v_i], "ns"))


def split_by_boundaries(ds: SequenceDataset, train_end: np.datetime64, validation_end: np.datetime64) -> ChronologicalSplit:
    """train: anchor <= train_end; validation: (train_end, validation_end]; test: > validation_end."""
    t_end = np.datetime64(train_end, "ns")
    v_end = np.datetime64(validation_end, "ns")
    if v_end < t_end:
        raise ValueError("validation_end must not precede train_end")
    d = ds.anchor_dates
    return ChronologicalSplit(
        train=ds.subset(d <= t_end),
        validation=ds.subset((d > t_end) & (d <= v_end)),
        test=ds.subset(d > v_end),
        train_end=str(pd.Timestamp(t_end).date()),
        validation_end=str(pd.Timestamp(v_end).date()),
    )
