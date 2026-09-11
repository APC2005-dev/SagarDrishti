"""Reproduce the base research artifact from the notebook recipe.

Use ONLY when the original ``global_gru_trajectory_model.keras`` /
``global_gru_trajectory_scalers.joblib`` are unavailable. The output is written
to a *separate* directory (default ``models/base-reproduced``) and labelled
``artifact_origin = reproduced_from_notebook_recipe``; it never overwrites the
original base. Copying it into ``models/base`` is a deliberate human decision.

    python -m ml.training.reproduce_base --zip data/bootstrap/consolidated_database_v8.0.zip \
        --out models/base-reproduced [--epochs 65]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib

from ml.adapters.bootstrap_adapter import DATASET_NAME, load_byu_tracks, sha256_file
from ml.constants import (
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    FEATURE_NAMES,
    FORECAST_DAYS,
    INPUT_SEMANTICS_RESEARCH,
    SEQUENCE_LENGTH,
)
from ml.evaluation.evaluator import evaluate_bundle, evaluate_constant_velocity
from ml.models.artifact_store import scaler_payload
from ml.models.gru_architecture import build_model
from ml.models.model_loader import ModelBundle
from ml.training.dataset_builder import build_historical_sequences, split_by_fraction
from ml.training.trainer import TrainingConfig, fit, fit_scalers, transform


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("models/base-reproduced"))
    p.add_argument("--epochs", type=int, default=65)
    args = p.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        print(f"refusing to overwrite non-empty {args.out}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    tracks = load_byu_tracks(args.zip)
    dataset = build_historical_sequences(tracks)
    split = split_by_fraction(dataset, 0.75, 0.125)
    print(json.dumps(split.summary(), indent=2))

    fs, ts = fit_scalers(split.train.X, split.train.y)
    Xtr, ytr = transform(fs, ts, split.train.X, split.train.y)
    Xva, yva = transform(fs, ts, split.validation.X, split.validation.y)
    model = build_model(ARCHITECTURE_VERSION)
    config = TrainingConfig(epochs=args.epochs)
    started = datetime.now(UTC)
    history = fit(model, Xtr, ytr, Xva, yva, config)

    bundle = ModelBundle("base-reproduced", model, fs, ts, {}, args.out)
    protocol = f"historical_test_{split.test.date_range()[0]}_{split.test.date_range()[1]}"
    result = evaluate_bundle(bundle, split.test, protocol)
    cv = evaluate_constant_velocity(split.test, protocol)

    model.save(args.out / "global_gru_trajectory_model.keras")
    joblib.dump(scaler_payload(fs, ts), args.out / "global_gru_trajectory_scalers.joblib")
    metadata = {
        "version": "base",
        "parent_version": None,
        "architecture": ARCHITECTURE,
        "architecture_version": ARCHITECTURE_VERSION,
        "input_sequence_length": SEQUENCE_LENGTH,
        "input_semantics": INPUT_SEMANTICS_RESEARCH,
        "forecast_horizon_days": FORECAST_DAYS,
        "feature_names": list(FEATURE_NAMES),
        "artifact_files": {"model": "global_gru_trajectory_model.keras", "scalers": "global_gru_trajectory_scalers.joblib"},
        "artifact_origin": "reproduced_from_notebook_recipe",
        "training": {
            "dataset": DATASET_NAME,
            "dataset_sha256": sha256_file(args.zip),
            "split": split.summary(),
            "config": config.as_dict(),
            "epochs_run": len(history.get("loss", [])),
            "started_at": started.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
        },
        "training_data_cutoff": str(tracks["date"].max().date()),
        "metrics": {"protocol": protocol, **result.as_dict(), "constant_velocity_benchmark": cv.as_dict()},
        "created_at": datetime.now(UTC).isoformat(),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
