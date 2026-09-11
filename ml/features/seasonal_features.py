"""Seasonal cycle encoding (notebook §3)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
from numpy.typing import NDArray

from ml.constants import DAYS_PER_YEAR


def day_of_year(dates: Sequence[date]) -> NDArray[np.float64]:
    return np.asarray([d.timetuple().tm_yday for d in dates], dtype=np.float64)


def season_features(dates: Sequence[date]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (season_sin, season_cos) with angle = 2*pi*day_of_year/365.25."""
    angle = 2 * np.pi * day_of_year(dates) / DAYS_PER_YEAR
    return np.sin(angle), np.cos(angle)
