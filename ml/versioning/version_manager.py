"""Model version identifiers: ``base`` then ``v1, v2, ... vN`` without upper bound."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

BASE_VERSION = "base"
_VERSION_RE = re.compile(r"^v([1-9][0-9]*)$")


def is_numbered(version: str) -> bool:
    return bool(_VERSION_RE.match(version))


def version_number(version: str) -> int:
    """``base`` -> 0, ``v12`` -> 12."""
    if version == BASE_VERSION:
        return 0
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(f"invalid model version {version!r}")
    return int(match.group(1))


def format_version(number: int) -> str:
    if number < 1:
        raise ValueError("numbered versions start at v1")
    return f"v{number}"


def next_version(existing: Iterable[str], models_root: Path | None = None) -> str:
    """Next free ``vN`` considering both the registry and directories on disk.

    Checking the filesystem too means a crashed run that left ``models/v9`` on
    disk (but never reached the registry) can never be overwritten.
    """
    numbers = [version_number(v) for v in existing if is_numbered(v)]
    if models_root is not None and Path(models_root).exists():
        numbers += [version_number(p.name) for p in Path(models_root).iterdir() if p.is_dir() and is_numbered(p.name)]
    return format_version(max(numbers, default=0) + 1)


# --- model families --------------------------------------------------------
# Each family owns an independent lineage: ``base`` then ``v1, v2, ... vN``.
# Trajectory v4 and sea-ice v4 are unrelated models that share a number only by
# coincidence.
FAMILY_TRAJECTORY = "trajectory"
FAMILY_SEA_ICE = "sea_ice"


def qualify(family: str, version: str) -> str:
    """Family-scoped identifier -> the globally unique registry key.

    Trajectory keeps its historical bare identifiers so existing rows and
    foreign keys are untouched; every other family is namespaced.
    """
    return version if family == FAMILY_TRAJECTORY else f"{family}/{version}"


def split_qualified(qualified: str) -> tuple[str, str]:
    """Inverse of :func:`qualify`: ``"sea_ice/v3"`` -> ``("sea_ice", "v3")``."""
    family, _, version = qualified.rpartition("/")
    return (family or FAMILY_TRAJECTORY, version)


def short_version(qualified: str) -> str:
    """The identifier as the family numbers it: ``"sea_ice/v3"`` -> ``"v3"``."""
    return split_qualified(qualified)[1]


def next_version_for_family(family: str, existing: Iterable[str], family_root: Path | None = None) -> str:
    """Next free ``vN`` **within one family**, from registry + filesystem state.

    ``existing`` may hold qualified or bare identifiers; entries belonging to
    other families are ignored, so families never consume each other's numbers.
    """
    mine = [short_version(v) for v in existing if split_qualified(v)[0] == family]
    return next_version(mine, family_root)
