"""Provider selection from configuration — the only place that maps
``ENV_*_PROVIDER`` families to concrete source adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ml.environment.providers.base import EnvironmentalProvider
from ml.environment.providers.copernicus_marine import SPECS as CMEMS_SPECS
from ml.environment.providers.copernicus_marine import CopernicusMarineProvider
from ml.environment.providers.era5 import SPECS as ERA5_SPECS
from ml.environment.providers.era5 import Era5Provider
from ml.environment.types import EnvGroup, ProviderRole, ProviderSpec

ALL_SPECS: dict[str, ProviderSpec] = {**CMEMS_SPECS, **ERA5_SPECS}

FAMILY_MAP: dict[tuple[EnvGroup, ProviderRole, str], str] = {
    (EnvGroup.WIND, ProviderRole.OPERATIONAL, "copernicus_marine"): "copernicus_marine_wind_nrt",
    (EnvGroup.WIND, ProviderRole.HISTORICAL, "copernicus_marine"): "copernicus_marine_wind_my",
    (EnvGroup.WIND, ProviderRole.HISTORICAL, "era5"): "era5_wind",
    (EnvGroup.CURRENT, ProviderRole.OPERATIONAL, "copernicus_marine"): "copernicus_marine_current_anfc",
    (EnvGroup.CURRENT, ProviderRole.HISTORICAL, "copernicus_marine"): "copernicus_marine_current_my",
    (EnvGroup.SEA_ICE, ProviderRole.OPERATIONAL, "copernicus_marine"): "copernicus_marine_sea_ice_anfc",
    (EnvGroup.SEA_ICE, ProviderRole.HISTORICAL, "copernicus_marine"): "copernicus_marine_sea_ice_my",
    (EnvGroup.SEA_ICE, ProviderRole.HISTORICAL, "era5"): "era5_sea_ice",
}


@dataclass(frozen=True)
class Credentials:
    copernicus_marine_username: str | None = None
    copernicus_marine_password: str | None = None
    cds_api_key: str | None = None
    cds_api_url: str | None = None


def provider_name(group: EnvGroup, role: ProviderRole, family: str) -> str:
    try:
        return FAMILY_MAP[(group, role, family)]
    except KeyError as exc:
        options = sorted({f for (g, r, f) in FAMILY_MAP if g == group and r == role})
        raise ValueError(f"no {role.value} {group.value} provider for family {family!r}; options: {options}") from exc


def build_provider(name: str, credentials: Credentials) -> EnvironmentalProvider:
    if name in CMEMS_SPECS:
        return CopernicusMarineProvider(
            CMEMS_SPECS[name], credentials.copernicus_marine_username, credentials.copernicus_marine_password
        )
    if name in ERA5_SPECS:
        return Era5Provider(ERA5_SPECS[name], credentials.cds_api_key, credentials.cds_api_url)
    raise ValueError(f"unknown provider {name!r}")


def build_role(
    families: Mapping[EnvGroup, str], role: ProviderRole, credentials: Credentials
) -> dict[EnvGroup, EnvironmentalProvider]:
    return {g: build_provider(provider_name(g, role, fam), credentials) for g, fam in families.items() if fam}
