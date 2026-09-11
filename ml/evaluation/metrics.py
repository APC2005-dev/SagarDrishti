"""Error summaries shared by operational evaluation and offline benchmarks.

"MAE" here is the mean great-circle distance between predicted and actual
positions (km) — the notebook's "mean error"; RMSE, median and p90 are computed
over the same distances.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ErrorSummary:
    n: int
    mae_km: float | None
    rmse_km: float | None
    median_km: float | None
    p90_km: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def summarize(errors_km: ArrayLike) -> ErrorSummary:
    e = np.asarray(errors_km, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return ErrorSummary(0, None, None, None, None)
    return ErrorSummary(
        n=int(e.size),
        mae_km=float(e.mean()),
        rmse_km=float(np.sqrt(np.mean(e**2))),
        median_km=float(np.median(e)),
        p90_km=float(np.quantile(e, 0.90)),
    )


def summarize_by_horizon(pairs: Iterable[tuple[int, float]]) -> dict[int, ErrorSummary]:
    grouped: dict[int, list[float]] = {}
    for horizon, err in pairs:
        grouped.setdefault(int(horizon), []).append(float(err))
    return {h: summarize(v) for h, v in sorted(grouped.items())}


def horizon_table(by_horizon: Mapping[int, ErrorSummary]) -> dict[str, dict[str, float | int | None]]:
    """JSON-friendly ``{"1": {...}, ..., "7": {...}}``."""
    return {str(h): s.as_dict() for h, s in by_horizon.items()}


def headline_errors(by_horizon: Mapping[int, ErrorSummary]) -> dict[str, float | None]:
    def mae(h: int) -> float | None:
        s = by_horizon.get(h)
        return s.mae_km if s else None

    return {"day1_error": mae(1), "day3_error": mae(3), "day7_error": mae(7)}
