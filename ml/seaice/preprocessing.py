"""Sea-ice preprocessing, reproducing the training notebook exactly.

Notebook pipeline, in order:

1. ``ice_frac = ds['ice_conc'] / 100.0``           (percent -> fraction)
2. ``.coarsen(latitude=5, longitude=5, boundary='trim').mean(skipna=True)``
3. ``conc = nan_to_num(block, nan=0.0)`` and ``mask = (~isnan(block))``
4. input = concat([conc_hist(7), mask_hist(7), doy_sin, doy_cos]) -> (16, H, W)

The seasonal channels use the day-of-year of the **first forecast target**
(``dates[t]`` in the notebook, where the window is ``conc[t-7:t]``). At training
time that is the real date of the h=1 target entry; at inference time the target
lies in the future, so it is the last window entry date + 1 day. Both are the
same quantity, and they coincide with the notebook on a gapless daily series.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from ml.seaice.constants import (
    COARSEN_BOUNDARY,
    COARSEN_FACTOR,
    DAYS_PER_YEAR,
    GRID_SHAPE,
    IN_CHANNELS,
    WINDOW,
)


class SeaIcePreprocessingError(ValueError):
    pass


def coarsen_ice_conc(data_array: object) -> tuple[np.ndarray, np.ndarray]:
    """xarray ``ice_conc`` (percent) -> ``(concentration, valid_mask)`` float32 grids.

    Returns the fraction grid with NaN replaced by 0, and the validity mask
    recording where the source actually had data. The mask is what keeps
    land and no-retrieval cells out of the loss and out of the metrics.
    """
    fraction = data_array / 100.0
    coarse = fraction.coarsen(
        latitude=COARSEN_FACTOR, longitude=COARSEN_FACTOR, boundary=COARSEN_BOUNDARY
    ).mean(skipna=True)
    values = np.asarray(coarse.values, dtype=np.float32)
    concentration = np.nan_to_num(values, nan=0.0).astype(np.float32)
    mask = (~np.isnan(values)).astype(np.float32)
    return concentration, mask


def day_of_year_encoding(day: date) -> tuple[float, float]:
    doy = day.timetuple().tm_yday
    return (
        float(np.sin(2 * np.pi * doy / DAYS_PER_YEAR)),
        float(np.cos(2 * np.pi * doy / DAYS_PER_YEAR)),
    )


def first_target_date(last_entry_date: date) -> date:
    """Date the h=1 horizon refers to: one day after the last window entry."""
    return last_entry_date + timedelta(days=1)


def build_input_tensor(
    concentration_history: np.ndarray, mask_history: np.ndarray, target_date: date
) -> np.ndarray:
    """``(WINDOW, H, W)`` history + target date -> ``(IN_CHANNELS, H, W)`` model input.

    ``concentration_history`` must be the 7 chronological entries in ascending
    date order; the last entry is the persistence field the residual is added to.
    """
    if concentration_history.shape != mask_history.shape:
        raise SeaIcePreprocessingError(
            f"concentration {concentration_history.shape} and mask {mask_history.shape} shapes differ"
        )
    if concentration_history.shape[0] != WINDOW:
        raise SeaIcePreprocessingError(
            f"expected {WINDOW} chronological entries, got {concentration_history.shape[0]}"
        )
    grid = tuple(concentration_history.shape[1:])
    if grid != GRID_SHAPE:
        raise SeaIcePreprocessingError(f"expected grid {GRID_SHAPE}, got {grid}")
    sin_v, cos_v = day_of_year_encoding(target_date)
    seasonal = np.concatenate(
        [
            np.full((1, *grid), sin_v, dtype=np.float32),
            np.full((1, *grid), cos_v, dtype=np.float32),
        ],
        axis=0,
    )
    tensor = np.concatenate(
        [concentration_history.astype(np.float32), mask_history.astype(np.float32), seasonal], axis=0
    )
    if tensor.shape[0] != IN_CHANNELS:
        raise SeaIcePreprocessingError(f"built {tensor.shape[0]} channels, expected {IN_CHANNELS}")
    return tensor
