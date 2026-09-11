"""Train a challenger from a champion bundle (pure, no database access).

Strategies
----------
``fine_tune`` (default)
    Start from the champion's weights and *keep its scalers*. Rescaling would
    silently change what the pretrained weights see, so fine-tuning never refits
    scalers. A replay sample of the historical training split is mixed with the
    operational samples to avoid forgetting the long daily record.

``from_scratch``
    Fresh weights from the architecture registry and scalers refitted on the new
    chronological training split only.

Either way the architecture comes from ``ml.models.gru_architecture`` keyed by
the parent's ``architecture_version`` — never an ad-hoc model definition.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ml.evaluation.evaluator import EvaluationResult, evaluate_bundle
from ml.models.gru_architecture import build_model
from ml.models.model_loader import ModelBundle
from ml.training.dataset_builder import SequenceDataset
from ml.training.trainer import TrainingConfig, fit, fit_scalers, transform

STRATEGY_FINE_TUNE = "fine_tune"
STRATEGY_FROM_SCRATCH = "from_scratch"


@dataclass(frozen=True)
class RetrainConfig:
    strategy: str = STRATEGY_FINE_TUNE
    historical_replay_samples: int = 50_000
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(learning_rate=1e-4, epochs=20))
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["training"] = self.training.as_dict()
        return d


@dataclass
class TrainedChallenger:
    bundle: ModelBundle  # in-memory bundle, not yet written to disk
    history: dict[str, list[float]]
    train_summary: dict[str, Any]
    validation_summary: dict[str, Any]
    started_at: float
    completed_at: float


def _replay(ds: SequenceDataset, n: int, seed: int) -> SequenceDataset:
    """Most-recent-biased deterministic sample of historical training windows."""
    if n <= 0 or len(ds) == 0:
        return SequenceDataset.empty()
    if len(ds) <= n:
        return ds
    rng = np.random.default_rng(seed)
    order = np.argsort(ds.anchor_dates, kind="mergesort")
    recent = order[-n // 2 :]
    rest = rng.choice(order[: -n // 2], size=n - len(recent), replace=False)
    mask = np.zeros(len(ds), dtype=bool)
    mask[np.concatenate([recent, rest])] = True
    return ds.subset(mask)


def train_challenger(
    champion: ModelBundle,
    historical_train: SequenceDataset,
    historical_validation: SequenceDataset,
    operational_train: SequenceDataset,
    operational_validation: SequenceDataset,
    config: RetrainConfig,
    candidate_version: str,
) -> TrainedChallenger:
    import keras

    started = time.time()
    architecture_version = champion.metadata.get("architecture_version")
    if not architecture_version:
        raise ValueError("champion metadata lacks architecture_version")

    if config.strategy == STRATEGY_FINE_TUNE:
        train = SequenceDataset.concat([_replay(historical_train, config.historical_replay_samples, config.seed), operational_train])
        validation = SequenceDataset.concat(
            [_replay(historical_validation, max(config.historical_replay_samples // 5, 1), config.seed), operational_validation]
        )
        model = build_model(architecture_version)
        model.set_weights(champion.model.get_weights())
        fs, ts = champion.feature_scaler, champion.target_scaler
    elif config.strategy == STRATEGY_FROM_SCRATCH:
        train = SequenceDataset.concat([historical_train, operational_train])
        validation = SequenceDataset.concat([historical_validation, operational_validation])
        keras.utils.set_random_seed(config.seed)
        model = build_model(architecture_version)
        fs, ts = fit_scalers(train.X, train.y)
    else:
        raise ValueError(f"unknown retraining strategy {config.strategy!r}")

    if len(train) == 0:
        raise ValueError("no training samples")
    Xtr, ytr = transform(fs, ts, train.X, train.y)
    Xva, yva = transform(fs, ts, validation.X, validation.y)
    history = fit(model, Xtr, ytr, Xva, yva, config.training)

    bundle = ModelBundle(
        version=candidate_version,
        model=model,
        feature_scaler=fs,
        target_scaler=ts,
        metadata={"architecture_version": architecture_version},
        directory=champion.directory,
    )
    return TrainedChallenger(
        bundle=bundle,
        history=history,
        train_summary=train.summary(),
        validation_summary=validation.summary(),
        started_at=started,
        completed_at=time.time(),
    )


def evaluate_pair(
    champion: ModelBundle, challenger: ModelBundle, dataset: SequenceDataset, protocol: str
) -> tuple[EvaluationResult, EvaluationResult]:
    """Same protocol, same samples, both models."""
    return evaluate_bundle(champion, dataset, protocol), evaluate_bundle(challenger, dataset, protocol)
