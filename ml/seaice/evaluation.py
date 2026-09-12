"""Masked sea-ice metrics, identical to the notebook's ``masked_rmse`` / ``masked_mae``.

Every metric is weighted by the validity mask, so cells the source never
retrieved (land, missing swaths) contribute nothing. This is the single
evaluation methodology used for the base model, for candidates and for the
champion, so their numbers are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.seaice.constants import HORIZONS


def masked_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    squared_error = ((pred - target) ** 2) * mask
    return float(np.sqrt(squared_error.sum() / max(float(mask.sum()), 1.0)))


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    absolute_error = np.abs(pred - target) * mask
    return float(absolute_error.sum() / max(float(mask.sum()), 1.0))


@dataclass
class HorizonAccumulator:
    """Streaming per-horizon sums, so an evaluation never has to fit in memory at once."""

    squared_error: np.ndarray = field(default_factory=lambda: np.zeros(len(HORIZONS)))
    absolute_error: np.ndarray = field(default_factory=lambda: np.zeros(len(HORIZONS)))
    mask_total: np.ndarray = field(default_factory=lambda: np.zeros(len(HORIZONS)))
    samples: int = 0

    def add(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        """``pred`` / ``target`` / ``mask`` shaped ``(len(HORIZONS), H, W)``."""
        for h in range(len(HORIZONS)):
            self.squared_error[h] += float((((pred[h] - target[h]) ** 2) * mask[h]).sum())
            self.absolute_error[h] += float((np.abs(pred[h] - target[h]) * mask[h]).sum())
            self.mask_total[h] += float(mask[h].sum())
        self.samples += 1

    def as_dict(self) -> dict[str, Any]:
        denom = np.maximum(self.mask_total, 1.0)
        rmse, mae = np.sqrt(self.squared_error / denom), self.absolute_error / denom
        overall_denom = max(float(self.mask_total.sum()), 1.0)
        return {
            "samples": self.samples,
            "by_horizon": {
                str(h): {"rmse": float(rmse[i]), "mae": float(mae[i]), "n_cells": int(self.mask_total[i])}
                for i, h in enumerate(HORIZONS)
            },
            "overall": {
                "rmse": float(np.sqrt(self.squared_error.sum() / overall_denom)),
                "mae": float(self.absolute_error.sum() / overall_denom),
                "n_cells": int(self.mask_total.sum()),
            },
        }
