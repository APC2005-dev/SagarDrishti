"""Model building, artifact immutability, loading, inference and training (TensorFlow)."""

from __future__ import annotations

import json
import os
import stat
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from ml.adapters.production_adapter import ProductionSequenceAdapter
from ml.constants import ARCHITECTURE_VERSION
from ml.provenance import Provenance
from ml.tests.conftest import make_track

pytestmark = pytest.mark.tf


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny model trained on synthetic data, written through the artifact store."""
    from ml.models.artifact_store import write_version
    from ml.models.gru_architecture import build_model
    from ml.training.trainer import TrainingConfig, fit, fit_scalers, transform

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 14, 6)).astype(np.float32)
    y = rng.normal(size=(64, 14)).astype(np.float32)
    fs, ts = fit_scalers(X, y)
    Xs, ys = transform(fs, ts, X, y)
    model = build_model()
    fit(model, Xs[:48], ys[:48], Xs[48:], ys[48:], TrainingConfig(epochs=1, batch_size=16))
    root = tmp_path_factory.mktemp("models")
    write_version(root, "v1", model, fs, ts, {"version": "v1", "architecture_version": ARCHITECTURE_VERSION})
    return root


def test_architecture_matches_base_artifact() -> None:
    from ml.models.gru_architecture import build_model

    model = build_model()
    assert model.count_params() == 44366  # original run's model.summary() (notebook §5)
    assert tuple(model.input_shape) == (None, 14, 6)
    assert tuple(model.output_shape) == (None, 14)
    names = [type(layer).__name__ for layer in model.layers]
    assert names == ["GRU", "LayerNormalization", "Dense", "Dropout", "Dense"]


def test_unknown_architecture() -> None:
    from ml.models.gru_architecture import build_model

    with pytest.raises(ValueError):
        build_model("transformer_v0")


def test_masked_loss_equals_huber_on_full_targets() -> None:
    import keras

    from ml.models.gru_architecture import masked_huber_loss

    rng = np.random.default_rng(1)
    yt = rng.normal(size=(8, 14)).astype(np.float32) * 3
    yp = rng.normal(size=(8, 14)).astype(np.float32)
    ref = float(keras.losses.Huber()(yt, yp))
    assert float(masked_huber_loss(yt, yp)) == pytest.approx(ref, rel=1e-5)
    yt_nan = yt.copy()
    yt_nan[:, :12] = np.nan
    val = float(masked_huber_loss(yt_nan, yp))
    assert np.isfinite(val)


def test_versions_are_immutable(trained_dir: Path) -> None:
    from ml.models.artifact_store import ArtifactExistsError, copy_version

    with pytest.raises(ArtifactExistsError):
        copy_version(trained_dir, trained_dir / "v1", "v1", {"version": "v1"})
    mode = os.stat(trained_dir / "v1" / "model.keras").st_mode
    assert not mode & stat.S_IWUSR


def test_load_bundle_and_checksum(trained_dir: Path) -> None:
    from ml.models.model_loader import load_model_bundle

    bundle = load_model_bundle(trained_dir / "v1")
    assert bundle.version == "v1"
    meta = json.loads((trained_dir / "v1" / "metadata.json").read_text())
    assert set(meta["artifact_sha256"]) == {"model.keras", "scalers.joblib"}


def test_tampered_artifact_detected(trained_dir: Path, tmp_path: Path) -> None:
    from ml.models.artifact_store import copy_version
    from ml.models.model_loader import ModelArtifactError, load_model_bundle

    copy_version(tmp_path, trained_dir / "v1", "v2", {"version": "v2", "architecture_version": ARCHITECTURE_VERSION})
    target = tmp_path / "v2" / "scalers.joblib"
    os.chmod(target, stat.S_IWUSR | stat.S_IRUSR)
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ModelArtifactError, match="checksum"):
        load_model_bundle(tmp_path / "v2")


def test_missing_artifact(tmp_path: Path) -> None:
    from ml.models.model_loader import ModelArtifactError, load_model_bundle

    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(ModelArtifactError):
        load_model_bundle(tmp_path)


def test_forecast_generation_single_pass_all_horizons(trained_dir: Path) -> None:
    from ml.inference.forecast_generator import filter_horizon, generate_forecasts
    from ml.models.model_loader import load_model_bundle

    bundle = load_model_bundle(trained_dir / "v1")
    hist = make_track("A81", date(2026, 3, 1), 30, 1, Provenance.HISTORICAL_TRAINING_DATASET)
    official = make_track("A81", date(2026, 9, 10), 1, 7, Provenance.OFFICIAL_USNIC, first_id=9)
    seq = ProductionSequenceAdapter().build(hist + official)
    [fc] = generate_forecasts(bundle, [seq], {1: 2.0, 3: 8.0, 7: 28.0})
    assert [p.horizon_days for p in fc.points] == list(range(1, 8))
    assert [p.forecast_date for p in fc.points] == [date(2026, 9, 10) + timedelta(days=h) for h in range(1, 8)]
    assert fc.points[0].risk_radius_km_p90 == 2.0 and fc.points[1].risk_radius_km_p90 is None
    assert all(-90 <= p.predicted_latitude <= 0 for p in fc.points)
    assert [p.horizon_days for p in filter_horizon(fc.points, 3)] == [1, 2, 3]
    with pytest.raises(ValueError):
        filter_horizon(fc.points, 8)


def test_evaluator_and_fine_tune(trained_dir: Path) -> None:
    import pandas as pd

    from ml.evaluation.evaluator import evaluate_bundle, evaluate_constant_velocity
    from ml.features.coordinate_transform import latlon_to_polar_m
    from ml.models.model_loader import load_model_bundle
    from ml.training.dataset_builder import build_historical_sequences, split_by_fraction
    from ml.training.retrainer import RetrainConfig, train_challenger
    from ml.training.trainer import TrainingConfig

    days = 80
    lat = -65 + np.arange(days) * 0.01
    lon = 40 + np.arange(days) * 0.03
    x, y = latlon_to_polar_m(lat, lon)
    tracks = pd.DataFrame(
        {"iceberg_id": "S1", "date": pd.date_range("2021-01-01", periods=days), "x_m": x, "y_m": y}
    )
    split = split_by_fraction(build_historical_sequences(tracks))
    champion = load_model_bundle(trained_dir / "v1")
    res = evaluate_bundle(champion, split.test, "unit")
    assert set(res.by_horizon) == set(range(1, 8)) and res.by_horizon[7].n == len(split.test)
    cv = evaluate_constant_velocity(split.test, "unit")
    assert cv.by_horizon[1].mae_km < 0.5  # straight-line track -> near-perfect benchmark
    challenger = train_challenger(
        champion,
        split.train,
        split.validation,
        split.train.subset(np.zeros(len(split.train), bool)),
        split.validation.subset(np.zeros(len(split.validation), bool)),
        RetrainConfig(training=TrainingConfig(epochs=1, batch_size=8), historical_replay_samples=20),
        "v2",
    )
    assert challenger.bundle.feature_scaler is champion.feature_scaler  # fine-tune keeps scalers
    assert challenger.train_summary["samples"] == 20
