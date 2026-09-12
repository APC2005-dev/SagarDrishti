"""Train a sea-ice challenger from the current champion's weights.

Mirrors the notebook's training loop: masked-RMSE loss on the residual
formulation, Adam, early stopping on validation RMSE, and the best epoch kept.
The split is chronological, never random.

Fine-tuning from the champion (rather than from scratch) is the default so a
candidate starts from the behaviour currently in production and the comparison
measures what the new data added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ml.seaice.architecture import SmallUNetResidual, build_model
from ml.seaice.dataset import SeaIceSample
from ml.seaice.inference import forecast


@dataclass
class RetrainConfig:
    epochs: int = 30
    learning_rate: float = 1e-4
    batch_size: int = 8
    patience: int = 5
    seed: int = 42
    initialisation: str = "champion_weights"

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "seed": self.seed,
            "initialisation": self.initialisation,
            "loss": "masked RMSE",
            "split": "chronological",
        }


@dataclass
class TrainingOutcome:
    model: SmallUNetResidual
    best_validation_rmse: float
    epochs_run: int
    train_samples: int
    validation_samples: int
    history: dict[str, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_validation_rmse": self.best_validation_rmse,
            "epochs_run": self.epochs_run,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "history": self.history,
        }


def _masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared_error = ((pred - target) ** 2) * mask
    return torch.sqrt(squared_error.sum() / mask.sum().clamp_min(1.0))


def _batches(samples: list[SeaIceSample], size: int, shuffle: bool, rng: np.random.Generator):
    order = np.arange(len(samples))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), size):
        chunk = [samples[i] for i in order[start : start + size]]
        yield (
            torch.from_numpy(np.stack([s.x for s in chunk])),
            torch.from_numpy(np.stack([s.y for s in chunk])),
            torch.from_numpy(np.stack([s.mask for s in chunk])),
        )


def evaluate_loss(model: SmallUNetResidual, samples: list[SeaIceSample], batch_size: int) -> float:
    if not samples:
        return float("inf")
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    with torch.no_grad():
        for xb, yb, mb in _batches(samples, batch_size, shuffle=False, rng=rng):
            losses.append(float(_masked_rmse(forecast(model, xb), yb, mb)))
    return float(np.mean(losses)) if losses else float("inf")


def train_challenger(
    train: list[SeaIceSample],
    validation: list[SeaIceSample],
    config: RetrainConfig,
    initial_state: dict[str, Any] | None = None,
) -> TrainingOutcome:
    """Fine-tune a challenger. ``initial_state`` is the champion's ``state_dict``."""
    if not train:
        raise ValueError("no training samples: cannot train a sea-ice challenger")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = build_model()
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_validation = evaluate_loss(model, validation, config.batch_size)
    history: dict[str, list[float]] = {"train_rmse": [], "validation_rmse": []}
    bad_epochs, epochs_run = 0, 0

    for _ in range(config.epochs):
        model.train()
        epoch_losses = []
        for xb, yb, mb in _batches(train, config.batch_size, shuffle=True, rng=rng):
            optimizer.zero_grad()
            loss = _masked_rmse(forecast(model, xb), yb, mb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        epochs_run += 1
        train_rmse = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        validation_rmse = evaluate_loss(model, validation, config.batch_size)
        history["train_rmse"].append(train_rmse)
        history["validation_rmse"].append(validation_rmse)
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    model.load_state_dict(best_state, strict=True)
    model.eval()
    return TrainingOutcome(
        model=model,
        best_validation_rmse=best_validation,
        epochs_run=epochs_run,
        train_samples=len(train),
        validation_samples=len(validation),
        history=history,
    )
