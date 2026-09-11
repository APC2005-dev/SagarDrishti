"""Model fitting (notebook §4–§5, generalised to NaN-masked targets)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from ml.models.gru_architecture import compile_model


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 5e-4
    epochs: int = 65
    batch_size: int = 512
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-5
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_scalers(X_train_raw: NDArray[np.float32], y_train_raw: NDArray[np.float32]) -> tuple[StandardScaler, StandardScaler]:
    """Fit on training data only. StandardScaler ignores NaN when fitting."""
    fs = StandardScaler().fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))  # any schema width
    ts = StandardScaler().fit(y_train_raw)
    return fs, ts


def transform(fs: StandardScaler, ts: StandardScaler, X: NDArray[np.float32], y: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    Xs = fs.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    ys = ts.transform(y).astype(np.float32)  # NaN stays NaN -> masked in the loss
    return Xs, ys


def fit(
    model: Any,
    X_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    config: TrainingConfig,
    checkpoint_path: Path | None = None,
) -> dict[str, list[float]]:
    """Train in place with the notebook's callbacks; returns the Keras history."""
    import keras
    from keras import callbacks

    keras.utils.set_random_seed(config.seed)
    compile_model(model, config.learning_rate)
    cbs: list[Any] = [
        callbacks.EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=config.reduce_lr_factor, patience=config.reduce_lr_patience, min_lr=config.min_learning_rate
        ),
    ]
    if checkpoint_path is not None:
        cbs.append(callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True))
    if len(X_val) == 0:
        raise ValueError("validation split is empty; refusing to train without a chronological validation set")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=cbs,
        shuffle=True,  # shuffles batches *within* the chronological training split only
        verbose=2,
    )
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}
