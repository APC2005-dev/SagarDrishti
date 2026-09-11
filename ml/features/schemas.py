"""Explicit feature schemas.

A model's input columns are defined by a *feature schema*, recorded with the
artifact (``metadata.json -> feature_schema_version / feature_names``) and in
the registry. Nothing is inferred from file names.

The six trajectory features come first in every schema and are computed
exactly as for the base model; environmental columns are appended. The base
artifact and v1 use ``trajectory_v1`` and never change.

Adding a variable: register an :class:`EnvVariable`, give it a provider group,
and add a schema that lists it. Existing schemas are frozen once a model has
been trained with them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ml.constants import FEATURE_NAMES as TRAJECTORY_FEATURES


@dataclass(frozen=True)
class EnvVariable:
    name: str
    group: str  # provider group: wind | current | sea_ice
    units: str
    description: str
    valid_range: tuple[float, float]


ENV_VARIABLES: dict[str, EnvVariable] = {
    v.name: v
    for v in (
        EnvVariable("wind_u", "wind", "m s-1", "eastward wind at 10 m, daily mean", (-60.0, 60.0)),
        EnvVariable("wind_v", "wind", "m s-1", "northward wind at 10 m, daily mean", (-60.0, 60.0)),
        EnvVariable("current_u", "current", "m s-1", "eastward sea-water velocity, uppermost model level (~0.5 m), daily mean", (-5.0, 5.0)),
        EnvVariable("current_v", "current", "m s-1", "northward sea-water velocity, uppermost model level (~0.5 m), daily mean", (-5.0, 5.0)),
        EnvVariable("sea_ice_concentration", "sea_ice", "1", "sea-ice area fraction (0-1), daily mean", (0.0, 1.0)),
    )
}
ALL_ENV_FEATURES: tuple[str, ...] = tuple(ENV_VARIABLES)
ALL_FEATURES: tuple[str, ...] = TRAJECTORY_FEATURES + ALL_ENV_FEATURES


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    description: str
    env_variables: tuple[str, ...]

    @property
    def features(self) -> tuple[str, ...]:
        return TRAJECTORY_FEATURES + self.env_variables

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def is_environmental(self) -> bool:
        return bool(self.env_variables)

    @property
    def groups(self) -> tuple[str, ...]:
        seen: list[str] = []
        for name in self.env_variables:
            g = ENV_VARIABLES[name].group
            if g not in seen:
                seen.append(g)
        return tuple(seen)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "description": self.description,
            "features": list(self.features),
            "env_variables": list(self.env_variables),
            "groups": list(self.groups),
        }


TRAJECTORY_SCHEMA = "trajectory_v1"
SCHEMAS: dict[str, FeatureSchema] = {
    s.version: s
    for s in (
        FeatureSchema(TRAJECTORY_SCHEMA, "trajectory only (base features)", ()),
        FeatureSchema("traj_wind_v1", "trajectory + wind", ("wind_u", "wind_v")),
        FeatureSchema("traj_current_v1", "trajectory + ocean current", ("current_u", "current_v")),
        FeatureSchema("traj_wind_current_v1", "trajectory + wind + ocean current", ("wind_u", "wind_v", "current_u", "current_v")),
        FeatureSchema(
            "traj_wind_current_ice_v1",
            "trajectory + wind + ocean current + sea ice",
            ("wind_u", "wind_v", "current_u", "current_v", "sea_ice_concentration"),
        ),
    )
}
for _s in SCHEMAS.values():
    for _v in _s.env_variables:
        assert _v in ENV_VARIABLES, f"schema {_s.version} references unknown variable {_v}"


def get_schema(version: str) -> FeatureSchema:
    try:
        return SCHEMAS[version]
    except KeyError as exc:
        raise ValueError(f"unknown feature schema {version!r}; known: {sorted(SCHEMAS)}") from exc


def schema_for_features(names: Sequence[str]) -> FeatureSchema:
    """Reverse lookup for metadata written before schemas were recorded (base, v1)."""
    names = tuple(names)
    for schema in SCHEMAS.values():
        if schema.features == names:
            return schema
    raise ValueError(f"feature list {names} matches no registered schema")


def column_indices(available: Sequence[str], wanted: Sequence[str]) -> list[int]:
    missing = [w for w in wanted if w not in available]
    if missing:
        raise ValueError(f"features {missing} not present in dataset columns {list(available)}")
    return [list(available).index(w) for w in wanted]


def select_columns(X: NDArray[np.float32], available: Sequence[str], wanted: Sequence[str]) -> NDArray[np.float32]:
    """Pick model columns (by name) from a wider feature tensor (..., n_available)."""
    idx = column_indices(available, wanted)
    if idx == list(range(len(available))):
        return X
    return X[..., idx]
