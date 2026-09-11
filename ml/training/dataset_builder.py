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

Columns are named (``feature_names``); the first six are always the trajectory
features. Environmental columns are appended by ``ml.environment.alignment``,
which needs the per-entry coordinates kept in ``entry_lat/lon/date``.

Splits are always chronological by anchor date — never shuffled — and every
sample from the same anchor date stays in the same split (notebook §4).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ml.adapters.production_adapter import ProductionSequenceAdapter
from ml.adapters.sequence_builder import ModelInputSequence, TrajectoryEntry
from ml.constants import FEATURE_NAMES, FORECAST_DAYS, N_FEATURES, OUTPUT_SIZE, SEQUENCE_LENGTH
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.features.schemas import select_columns
from ml.features.trajectory_features import build_target_vector
from ml.provenance import Provenance

SAMPLE_HISTORICAL = "historical"
SAMPLE_OPERATIONAL = "operational"


@dataclass
class SequenceDataset:
    X: NDArray[np.float32]  # (n, 14, n_features) unscaled
    y: NDArray[np.float32]  # (n, 14) km displacement, NaN = unknown horizon
    anchor_dates: NDArray[np.datetime64]
    anchor_x_m: NDArray[np.float64]
    anchor_y_m: NDArray[np.float64]
    iceberg_ids: NDArray[np.object_]
    sample_kind: NDArray[np.object_]
    anchor_observation_ids: NDArray[np.object_] = field(default_factory=lambda: np.zeros(0, dtype=object))
    feature_names: tuple[str, ...] = FEATURE_NAMES
    # Per-entry coordinates (n, 14); present when built with with_entries=True / for operational samples.
    entry_lat: NDArray[np.float32] | None = None
    entry_lon: NDArray[np.float32] | None = None
    entry_date: NDArray[np.datetime64] | None = None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @classmethod
    def empty(cls, feature_names: tuple[str, ...] = FEATURE_NAMES) -> SequenceDataset:
        return cls(
            X=np.zeros((0, SEQUENCE_LENGTH, len(feature_names)), np.float32),
            y=np.zeros((0, OUTPUT_SIZE), np.float32),
            anchor_dates=np.zeros(0, "datetime64[ns]"),
            anchor_x_m=np.zeros(0),
            anchor_y_m=np.zeros(0),
            iceberg_ids=np.zeros(0, object),
            sample_kind=np.zeros(0, object),
            anchor_observation_ids=np.zeros(0, object),
            feature_names=feature_names,
            entry_lat=np.zeros((0, SEQUENCE_LENGTH), np.float32),
            entry_lon=np.zeros((0, SEQUENCE_LENGTH), np.float32),
            entry_date=np.zeros((0, SEQUENCE_LENGTH), "datetime64[D]"),
        )

    def _ids(self) -> NDArray[np.object_]:
        return self.anchor_observation_ids if len(self.anchor_observation_ids) == len(self) else np.full(len(self), None, object)

    @property
    def has_entries(self) -> bool:
        return self.entry_lat is not None and self.entry_lon is not None and self.entry_date is not None

    def subset(self, mask: NDArray[np.bool_] | NDArray[np.int_]) -> SequenceDataset:
        return SequenceDataset(
            X=self.X[mask],
            y=self.y[mask],
            anchor_dates=self.anchor_dates[mask],
            anchor_x_m=self.anchor_x_m[mask],
            anchor_y_m=self.anchor_y_m[mask],
            iceberg_ids=self.iceberg_ids[mask],
            sample_kind=self.sample_kind[mask],
            anchor_observation_ids=self._ids()[mask],
            feature_names=self.feature_names,
            entry_lat=self.entry_lat[mask] if self.entry_lat is not None else None,
            entry_lon=self.entry_lon[mask] if self.entry_lon is not None else None,
            entry_date=self.entry_date[mask] if self.entry_date is not None else None,
        )

    def select(self, names: Sequence[str]) -> NDArray[np.float32]:
        """Feature tensor restricted to ``names`` (in that order)."""
        return np.ascontiguousarray(select_columns(self.X, self.feature_names, names))

    def with_features(self, X: NDArray[np.float32], names: Sequence[str]) -> SequenceDataset:
        if X.shape[:2] != self.X.shape[:2] or X.shape[2] != len(names):
            raise ValueError("feature tensor does not match dataset samples / names")
        return replace(self, X=X, feature_names=tuple(names))

    @staticmethod
    def concat(parts: Iterable[SequenceDataset]) -> SequenceDataset:
        all_parts = list(parts)
        parts = [p for p in all_parts if len(p)]
        if not parts:
            return SequenceDataset.empty(all_parts[0].feature_names if all_parts else FEATURE_NAMES)
        names = parts[0].feature_names
        if any(p.feature_names != names for p in parts):
            raise ValueError("cannot concatenate datasets with different feature columns")
        with_entries = all(p.has_entries for p in parts)
        return SequenceDataset(
            X=np.concatenate([p.X for p in parts]),
            y=np.concatenate([p.y for p in parts]),
            anchor_dates=np.concatenate([p.anchor_dates for p in parts]),
            anchor_x_m=np.concatenate([p.anchor_x_m for p in parts]),
            anchor_y_m=np.concatenate([p.anchor_y_m for p in parts]),
            iceberg_ids=np.concatenate([p.iceberg_ids for p in parts]),
            sample_kind=np.concatenate([p.sample_kind for p in parts]),
            anchor_observation_ids=np.concatenate([p._ids() for p in parts]),
            feature_names=names,
            entry_lat=np.concatenate([p.entry_lat for p in parts]) if with_entries else None,  # type: ignore[misc]
            entry_lon=np.concatenate([p.entry_lon for p in parts]) if with_entries else None,  # type: ignore[misc]
            entry_date=np.concatenate([p.entry_date for p in parts]) if with_entries else None,  # type: ignore[misc]
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
            "feature_names": list(self.feature_names),
        }


# --------------------------------------------------------------------------- historical
def build_historical_sequences(tracks: pd.DataFrame, with_entries: bool = False) -> SequenceDataset:
    """Notebook §3: 14-day inputs -> 7-day targets inside daily runs.

    ``tracks`` needs iceberg_id, date (datetime64), x_m, y_m (and latitude,
    longitude when ``with_entries``).
    """
    X_list: list[NDArray[np.float32]] = []
    y_list: list[NDArray[np.float32]] = []
    dates, ax, ay, ids = [], [], [], []
    e_lat: list[NDArray[np.float32]] = []
    e_lon: list[NDArray[np.float32]] = []
    e_date: list[NDArray[np.datetime64]] = []
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
            if with_entries:
                lat_all = run["latitude"].to_numpy(np.float32)
                lon_all = run["longitude"].to_numpy(np.float32)
                d_day = d.astype("datetime64[D]")
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
                if with_entries:
                    e_lat.append(lat_all[start : end + 1])
                    e_lon.append(lon_all[start : end + 1])
                    e_date.append(d_day[start : end + 1])
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
        entry_lat=np.stack(e_lat) if with_entries else None,
        entry_lon=np.stack(e_lon) if with_entries else None,
        entry_date=np.stack(e_date) if with_entries else None,
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
    e_lat, e_lon, e_date = [], [], []
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
            e_lat.append(np.asarray([e.latitude for e in seq.entries], np.float32))
            e_lon.append(np.asarray([e.longitude for e in seq.entries], np.float32))
            e_date.append(np.asarray([e.observation_date for e in seq.entries], dtype="datetime64[D]"))
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
        entry_lat=np.stack(e_lat),
        entry_lon=np.stack(e_lon),
        entry_date=np.stack(e_date),
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
        e = SequenceDataset.empty(ds.feature_names)
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


def stratified_cap(ds: SequenceDataset, max_samples: int, seed: int = 42) -> SequenceDataset:
    """Deterministic, chronologically even subsample (keeps date spread; order preserved)."""
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    order = np.argsort(ds.anchor_dates, kind="mergesort")
    rng = np.random.default_rng(seed)
    buckets = np.array_split(order, max_samples)
    picks = np.sort(np.array([b[rng.integers(len(b))] for b in buckets if len(b)]))
    return ds.subset(picks)


__all__ = [
    "N_FEATURES",
    "SAMPLE_HISTORICAL",
    "SAMPLE_OPERATIONAL",
    "ChronologicalSplit",
    "SequenceDataset",
    "build_historical_sequences",
    "build_operational_samples",
    "split_by_boundaries",
    "split_by_fraction",
    "stratified_cap",
]
