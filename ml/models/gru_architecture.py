"""Architecture registry.

``gru_entry14_to_day7_v1`` is the base research architecture, copied layer for
layer from the research notebook §5 (the code that produced
``global_gru_trajectory_model.keras``)::

    Input(14, 6)
    GRU(96, dropout=0.10, name="trajectory_gru")
    LayerNormalization()
    Dense(128, activation="relu")
    Dropout(0.15)
    Dense(14, name="future_displacements")

NOTE: the written project brief lists the stack without the LayerNormalization
layer, but the trained artifact contains it (192 params in the notebook's model
summary). The artifact is authoritative, so the layer stays. Removing it would
be a new architecture version, not v1.

New architectures must be added here under a new key; existing keys are frozen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ml.constants import ARCHITECTURE_VERSION, N_FEATURES, OUTPUT_SIZE, SEQUENCE_LENGTH

if TYPE_CHECKING:
    import keras


def _build_gru_entry14_to_day7_v1() -> keras.Model:
    import keras
    from keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(SEQUENCE_LENGTH, N_FEATURES)),
            layers.GRU(96, dropout=0.10, name="trajectory_gru"),
            layers.LayerNormalization(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(OUTPUT_SIZE, name="future_displacements"),
        ]
    )


ARCHITECTURES: dict[str, Callable[[], keras.Model]] = {
    ARCHITECTURE_VERSION: _build_gru_entry14_to_day7_v1,
}


def build_model(architecture_version: str = ARCHITECTURE_VERSION) -> keras.Model:
    try:
        builder = ARCHITECTURES[architecture_version]
    except KeyError as exc:
        raise ValueError(f"unknown architecture_version {architecture_version!r}") from exc
    return builder()


def masked_huber_loss(y_true, y_pred):  # type: ignore[no-untyped-def]
    """Huber loss that ignores NaN targets.

    Operational samples only know the horizons where an official observation
    actually exists (typically D+7 for a weekly product); missing horizons are
    NaN. With a fully observed target this equals ``keras.losses.Huber()``
    (mean over all elements), so historical and operational samples share one
    objective. The architecture is unchanged — only the training objective
    tolerates missing labels.
    """
    from keras import ops

    mask = ops.logical_not(ops.isnan(y_true))
    safe_true = ops.where(mask, y_true, ops.zeros_like(y_true))
    safe_pred = ops.where(mask, y_pred, ops.zeros_like(y_pred))
    error = safe_true - safe_pred
    abs_err = ops.abs(error)
    quadratic = ops.minimum(abs_err, 1.0)
    linear = abs_err - quadratic
    per_elem = 0.5 * ops.square(quadratic) + linear
    maskf = ops.cast(mask, per_elem.dtype)
    denom = ops.maximum(ops.sum(maskf), 1.0)
    return ops.sum(per_elem * maskf) / denom


def masked_mae(y_true, y_pred):  # type: ignore[no-untyped-def]
    from keras import ops

    mask = ops.logical_not(ops.isnan(y_true))
    diff = ops.where(mask, ops.abs(y_true - y_pred), ops.zeros_like(y_pred))
    maskf = ops.cast(mask, diff.dtype)
    return ops.sum(diff) / ops.maximum(ops.sum(maskf), 1.0)


def compile_model(model: keras.Model, learning_rate: float = 5e-4) -> keras.Model:
    """Compile with the notebook optimiser (Adam) and a NaN-tolerant Huber loss."""
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_huber_loss,
        metrics=[masked_mae],
    )
    return model
