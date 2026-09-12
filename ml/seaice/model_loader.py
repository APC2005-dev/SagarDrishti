"""Load a sea-ice version from disk and verify it against its recorded identity.

The ``.pt`` files are ``state_dict`` saves (that is how the notebook wrote the
base model), so the architecture is rebuilt from :mod:`ml.seaice.architecture`
and the weights are loaded with ``strict=True``. A checksum that no longer
matches, or a state dict whose channel counts disagree with the notebook's
tensor contract, refuses to load rather than silently serving a different model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ml.adapters.bootstrap_adapter import sha256_file
from ml.seaice.architecture import SmallUNetResidual, build_model
from ml.seaice.artifact_store import resolve_model_path
from ml.seaice.constants import (
    ARCHITECTURE_VERSION,
    BASE_CHANNELS,
    IN_CHANNELS,
    OUT_CHANNELS,
)


class SeaIceArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeaIceBundle:
    model: SmallUNetResidual
    version: str | None
    metadata: dict[str, Any]
    checksums: dict[str, str]
    model_path: Path


def read_metadata(directory: Path) -> dict[str, Any]:
    path = Path(directory) / "metadata.json"
    if not path.exists():
        raise SeaIceArtifactError(f"missing metadata.json in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_state_dict(state_dict: dict[str, Any]) -> None:
    """Check the saved tensors against the notebook's channel contract before loading."""
    for key in ("e1.0.weight", "out.weight"):
        if key not in state_dict:
            raise SeaIceArtifactError(f"state dict is missing {key}; not a {ARCHITECTURE_VERSION} artifact")
    in_ch = int(state_dict["e1.0.weight"].shape[1])
    base = int(state_dict["e1.0.weight"].shape[0])
    out_ch = int(state_dict["out.weight"].shape[0])
    if (in_ch, out_ch, base) != (IN_CHANNELS, OUT_CHANNELS, BASE_CHANNELS):
        raise SeaIceArtifactError(
            f"tensor contract mismatch: artifact is in_ch={in_ch} out_ch={out_ch} base={base}, "
            f"expected in_ch={IN_CHANNELS} out_ch={OUT_CHANNELS} base={BASE_CHANNELS}"
        )


def load_seaice_bundle(directory: Path, version: str | None = None, verify: bool = True) -> SeaIceBundle:
    directory = Path(directory)
    model_path = resolve_model_path(directory)
    metadata = read_metadata(directory)
    checksums = {model_path.name: sha256_file(model_path)}
    if verify:
        recorded = metadata.get("artifact_sha256") or {}
        for name, digest in recorded.items():
            actual = checksums.get(name)
            if actual is not None and actual != digest:
                raise SeaIceArtifactError(f"{directory.name}: {name} differs from the recorded checksum")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    validate_state_dict(state_dict)
    model = build_model()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return SeaIceBundle(
        model=model,
        version=version or metadata.get("version"),
        metadata=metadata,
        checksums=checksums,
        model_path=model_path,
    )
