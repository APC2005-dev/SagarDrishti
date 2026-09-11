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
summary). The artifact is authoritative, so the layer stays. It is fixed at six
input features and is never trained with environmental inputs.

``gru_env_concat_entry14_to_day7_v1`` is the first environmental architecture:
the *same* layer stack with ``Input(14, N)``, N = 6 trajectory features +
environmental columns concatenated per entry. Keeping the stack identical
isolates the question "does environmental information help?" from
architecture changes.

Extension point (not implemented until the concatenated model is established):
a two-branch model — trajectory GRU → embedding, environmental encoder →
embedding, fusion → decoder — would be registered here under a new key such
as ``gru_env_twobranch_entry14_to_day7_v1``, taking the same (14, N) input and
splitting columns by feature schema. Existing keys are frozen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ml.constants import ARCHITECTURE_VERSION, N_FEATURES, OUTPUT_SIZE, SEQUENCE_LENGTH

if TYPE_CHECKING:
    import keras

ENV_CONCAT_ARCHITECTURE_VERSION = "gru_env_concat_entry14_to_day7_v1"


def _gru_stack(n_features: int) -> keras.Model:
    import keras
    from keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
            layers.GRU(96, dropout=0.10, name="trajectory_gru"),
            layers.LayerNormalization(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(OUTPUT_SIZE, name="future_displacements"),
        ]
    )


def _build_gru_entry14_to_day7_v1(n_features: int = N_FEATURES) -> keras.Model:
    if n_features != N_FEATURES:
        raise ValueError(f"{ARCHITECTURE_VERSION} is fixed at {N_FEATURES} trajectory features (got {n_features})")
    return _gru_stack(N_FEATURES)


def _build_gru_env_concat_entry14_to_day7_v1(n_features: int) -> keras.Model:
    if n_features <= N_FEATURES:
        raise ValueError(f"{ENV_CONCAT_ARCHITECTURE_VERSION} needs trajectory + environmental features (> {N_FEATURES})")
    return _gru_stack(n_features)


ARCHITECTURES: dict[str, Callable[..., keras.Model]] = {
    ARCHITECTURE_VERSION: _build_gru_entry14_to_day7_v1,
    ENV_CONCAT_ARCHITECTURE_VERSION: _build_gru_env_concat_entry14_to_day7_v1,
}


def build_model(architecture_version: str = ARCHITECTURE_VERSION, n_features: int | None = None) -> keras.Model:
    try:
        builder = ARCHITECTURES[architecture_version]
    except KeyError as exc:
        raise ValueError(f"unknown architecture_version {architecture_version!r}") from exc
    return builder() if n_features is None else builder(n_features)


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
