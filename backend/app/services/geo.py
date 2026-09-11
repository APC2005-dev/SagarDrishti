"""Small helpers for PostGIS values. Storage SRID is always 4326."""

from __future__ import annotations


def ewkt_point(latitude: float, longitude: float) -> str:
    """EWKT for a WGS84 point. Note PostGIS axis order is (lon lat)."""
    return f"SRID=4326;POINT({longitude:.8f} {latitude:.8f})"
