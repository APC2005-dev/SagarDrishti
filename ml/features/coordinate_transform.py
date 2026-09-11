"""Explicit EPSG:4326 <-> EPSG:3031 conversion and geodesic distance.

Storage is always WGS84 (EPSG:4326). The model works in Antarctic Polar
Stereographic metres (EPSG:3031). Every conversion goes through this module so
coordinate systems are never mixed implicitly. ``always_xy=True`` means pyproj
takes/returns (lon, lat) — the functions below expose a (lat, lon) API and do
the swap in one place.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyproj import Transformer

from ml.constants import EARTH_RADIUS_KM, GEOGRAPHIC_CRS, PROJECTED_CRS


@lru_cache(maxsize=1)
def _to_polar() -> Transformer:
    return Transformer.from_crs(GEOGRAPHIC_CRS, PROJECTED_CRS, always_xy=True)


@lru_cache(maxsize=1)
def _to_geographic() -> Transformer:
    return Transformer.from_crs(PROJECTED_CRS, GEOGRAPHIC_CRS, always_xy=True)


def latlon_to_polar_m(
    latitude: ArrayLike, longitude: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """WGS84 degrees -> EPSG:3031 metres. Returns (x_m, y_m)."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    x, y = _to_polar().transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def polar_m_to_latlon(
    x_m: ArrayLike, y_m: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """EPSG:3031 metres -> WGS84 degrees. Returns (latitude, longitude)."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    lon, lat = _to_geographic().transform(x, y)
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def haversine_km(
    lat1: ArrayLike, lon1: ArrayLike, lat2: ArrayLike, lon2: ArrayLike
) -> NDArray[np.float64]:
    """Great-circle distance in km (identical formula to notebook §6)."""
    p1 = np.radians(np.asarray(lat1, dtype=np.float64))
    l1 = np.radians(np.asarray(lon1, dtype=np.float64))
    p2 = np.radians(np.asarray(lat2, dtype=np.float64))
    l2 = np.radians(np.asarray(lon2, dtype=np.float64))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
