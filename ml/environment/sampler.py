"""Point sampling of canonical daily grids.

Interpolation methodology (recorded as ``bilinear_nan_aware_daily``):

* **time** — observation dates carry no time of day, so the value for date *d*
  is the field for UTC day *d* (providers deliver daily means: hourly sources
  are averaged over [00:00, 24:00) UTC). No interpolation across days.
* **space** — bilinear interpolation between the four grid nodes surrounding
  the point. Nodes without data (land, ice shelf, outside the source mask) are
  dropped and the remaining weights renormalised; ``valid_neighbours`` records
  how many of the four were used. If none is valid the value is missing —
  nothing is extrapolated or invented.
* Points outside the grid extent are missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

INTERPOLATION = "bilinear_nan_aware_daily"


@dataclass(frozen=True)
class PointValue:
    values: dict[str, float | None]
    valid_neighbours: int
    reason: str | None


def _bracket(axis: np.ndarray, x: float) -> tuple[int, int, float] | None:
    """Indices (i0, i1) around x and fractional weight of i1; None if outside."""
    if x < axis[0] or x > axis[-1]:
        return None
    i1 = int(np.searchsorted(axis, x, side="left"))
    if i1 == 0:
        return 0, 0, 0.0
    if i1 >= len(axis):
        i1 = len(axis) - 1
    i0 = i1 - 1
    span = axis[i1] - axis[i0]
    w = 0.0 if span == 0 else float((x - axis[i0]) / span)
    return i0, i1, w


def sample_point(ds, variables: Sequence[str], lat: float, lon: float, day: date) -> PointValue:  # type: ignore[no-untyped-def]
    """Sample ``variables`` of a canonical daily dataset at (lat, lon) on ``day``."""
    times = ds["time"].values.astype("datetime64[D]")
    hit = np.nonzero(times == np.datetime64(day, "D"))[0]
    if hit.size == 0:
        return PointValue({v: None for v in variables}, 0, f"no field for {day}")
    t = int(hit[0])
    lats = ds["lat"].values.astype(np.float64)
    lons = ds["lon"].values.astype(np.float64)
    bl, bo = _bracket(lats, lat), _bracket(lons, lon)
    if bl is None or bo is None:
        return PointValue({v: None for v in variables}, 0, "point outside grid")
    i0, i1, wy = bl
    j0, j1, wx = bo
    corners = [(i0, j0, (1 - wy) * (1 - wx)), (i0, j1, (1 - wy) * wx), (i1, j0, wy * (1 - wx)), (i1, j1, wy * wx)]

    out: dict[str, float | None] = {}
    n_valid_min = 4
    for v in variables:
        arr = ds[v].values  # (time, lat, lon)
        num = 0.0
        den = 0.0
        n_valid = 0
        for i, j, w in corners:
            val = float(arr[t, i, j])
            if np.isfinite(val):
                num += w * val
                den += w
                n_valid += 1
        if n_valid == 0:
            out[v] = None
        elif den == 0.0:  # point sits exactly on valid nodes with zero weight elsewhere
            finite = [float(arr[t, i, j]) for i, j, _ in corners if np.isfinite(arr[t, i, j])]
            out[v] = float(np.mean(finite))
        else:
            out[v] = num / den
        n_valid_min = min(n_valid_min, n_valid)
    reason = None if all(val is not None for val in out.values()) else "no valid grid node around point"
    return PointValue(out, n_valid_min, reason)
