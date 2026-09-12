"""Geographic coordinates of the coarsened sea-ice grid.

The source is a regular 0.1 degree latitude/longitude grid (EPSG:4326) running
from -85.0 to -35.1 in latitude and -180.0 to 179.9 in longitude, verified
against the live catalogue. ``coarsen(latitude=5, longitude=5, boundary="trim")``
averages each 5x5 block, so a coarse cell's centre is the mean of the five
native coordinates it covers:

    centre = first + (5 * index + 2) * 0.1

which is ``first + 0.5 * index + 0.2``. These are derived from the preprocessing,
not measured or guessed, so the overlay lines up with the field the model
actually consumed.
"""

from __future__ import annotations

import numpy as np

from ml.seaice.constants import (
    COARSEN_FACTOR,
    GRID_SHAPE,
    NATIVE_LAT_RANGE,
    NATIVE_LON_RANGE,
    NATIVE_RESOLUTION_DEG,
)

# Standard sea-ice *extent* threshold: below 15% concentration a cell is
# conventionally treated as open water rather than ice.
ICE_EDGE_THRESHOLD = 0.15


def _centres(first: float, count: int) -> np.ndarray:
    offset = (COARSEN_FACTOR - 1) / 2.0 * NATIVE_RESOLUTION_DEG
    step = COARSEN_FACTOR * NATIVE_RESOLUTION_DEG
    return (first + offset + step * np.arange(count)).astype(np.float64)


def latitudes() -> np.ndarray:
    return _centres(NATIVE_LAT_RANGE[0], GRID_SHAPE[0])


def longitudes() -> np.ndarray:
    return _centres(NATIVE_LON_RANGE[0], GRID_SHAPE[1])


def to_points(
    field: np.ndarray,
    mask: np.ndarray | None = None,
    min_concentration: float = ICE_EDGE_THRESHOLD,
    stride: int = 1,
) -> list[list[float]]:
    """``(H, W)`` grid -> ``[[lat, lon, concentration], ...]`` for map rendering.

    Only cells the source actually retrieved (``mask``) and at or above
    ``min_concentration`` are emitted: open ocean carries no signal to draw and
    would multiply the payload for nothing. Nothing is interpolated or filled.
    """
    if field.shape != GRID_SHAPE:
        raise ValueError(f"expected grid {GRID_SHAPE}, got {field.shape}")
    stride = max(1, int(stride))
    lats, lons = latitudes()[::stride], longitudes()[::stride]
    values = field[::stride, ::stride]
    valid = (mask[::stride, ::stride] > 0) if mask is not None else np.ones_like(values, dtype=bool)
    rows, cols = np.nonzero(valid & (values >= min_concentration))
    return [
        [round(float(lats[r]), 4), round(float(lons[c]), 4), round(float(values[r, c]), 4)]
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True)
    ]
