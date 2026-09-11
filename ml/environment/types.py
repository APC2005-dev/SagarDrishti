"""Shared environmental data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum


class EnvGroup(StrEnum):
    WIND = "wind"
    CURRENT = "current"
    SEA_ICE = "sea_ice"


GROUP_VARIABLES: dict[EnvGroup, tuple[str, ...]] = {
    EnvGroup.WIND: ("wind_u", "wind_v"),
    EnvGroup.CURRENT: ("current_u", "current_v"),
    EnvGroup.SEA_ICE: ("sea_ice_concentration",),
}


class ProviderRole(StrEnum):
    OPERATIONAL = "operational"  # near-real-time analysis used for live forecasts
    HISTORICAL = "historical"  # reanalysis / reprocessed record used to build training features


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self) -> None:
        if not (-90 <= self.lat_min < self.lat_max <= 90 and -180 <= self.lon_min < self.lon_max <= 180):
            raise ValueError(f"invalid bbox {self}")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of one environmental source (what the UI and metadata report)."""

    name: str  # registry key, e.g. "copernicus_marine_wind_nrt"
    group: EnvGroup
    role: ProviderRole
    authority: str  # e.g. "Copernicus Marine Service"
    product_id: str
    dataset_id: str
    native_variables: dict[str, str]  # canonical name -> variable name in the source
    units: dict[str, str]
    spatial_resolution_deg: float
    temporal_resolution: str  # native: "PT1H" | "P1D"
    aggregation: str  # how native steps become one value per UTC day
    latency_days: float  # typical delay between a day and its availability
    coverage_start: date
    coverage_end: date | None = None  # None = ongoing
    depth: str | None = None  # documented vertical level for ocean fields
    supports_forecast: bool = False
    forecast_lead_days: int = 0
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["group"] = self.group.value
        d["role"] = self.role.value
        d["coverage_start"] = self.coverage_start.isoformat()
        d["coverage_end"] = self.coverage_end.isoformat() if self.coverage_end else None
        return d


@dataclass(frozen=True)
class EnvSample:
    """One environmental value set for one trajectory entry, with full provenance."""

    group: EnvGroup
    values: dict[str, float | None]
    requested_date: date  # the trajectory entry's observation date
    as_of: date  # prediction time T — nothing valid after the availability limit at T is used
    valid_date: date | None  # date of the field actually used
    provider: str
    dataset_id: str
    interpolation: str
    valid_neighbours: int
    missing: bool
    reason: str | None = None
    cache_key: str | None = None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def staleness_days(self) -> int | None:
        return None if self.valid_date is None else (self.requested_date - self.valid_date).days

    def as_dict(self) -> dict[str, object]:
        return {
            "group": self.group.value,
            "values": self.values,
            "requested_date": self.requested_date.isoformat(),
            "as_of": self.as_of.isoformat(),
            "valid_date": self.valid_date.isoformat() if self.valid_date else None,
            "staleness_days": self.staleness_days,
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "interpolation": self.interpolation,
            "valid_neighbours": self.valid_neighbours,
            "missing": self.missing,
            "reason": self.reason,
            "cache_key": self.cache_key,
            "quality_flags": list(self.quality_flags),
        }
