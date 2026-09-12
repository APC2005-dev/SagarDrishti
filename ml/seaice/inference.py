"""Sea-ice forecasting: the residual formulation from the notebook.

``pred = clamp(persistence + model(x), 0, 1)`` where ``persistence`` is the last
concentration entry of the input window (channel ``PERSISTENCE_CHANNEL``). The
network outputs a *delta*, never an absolute concentration, so this
postprocessing is mandatory: treating the raw network output as a forecast would
be wrong.
"""

from __future__ import annotations

import numpy as np
import torch

from ml.seaice.constants import HORIZONS, PERSISTENCE_CHANNEL


def persistence_field(x: torch.Tensor) -> torch.Tensor:
    """The last concentration entry, shaped ``(B, 1, H, W)`` so it broadcasts over horizons."""
    return x[:, PERSISTENCE_CHANNEL : PERSISTENCE_CHANNEL + 1, :, :]


def forecast(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``(B, IN_CH, H, W)`` -> ``(B, len(HORIZONS), H, W)`` concentration in [0, 1]."""
    return torch.clamp(persistence_field(x) + model(x), 0, 1)


def forecast_numpy(model: torch.nn.Module, tensor: np.ndarray) -> np.ndarray:
    """Single sample: ``(IN_CH, H, W)`` -> ``(len(HORIZONS), H, W)``."""
    model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)).unsqueeze(0)
        return forecast(model, batch).squeeze(0).cpu().numpy()


def persistence_forecast_numpy(tensor: np.ndarray) -> np.ndarray:
    """The no-skill baseline the notebook compares against: repeat the last entry."""
    last = tensor[PERSISTENCE_CHANNEL]
    return np.stack([last] * len(HORIZONS), axis=0).astype(np.float32)
