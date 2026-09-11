"""Environmental candidate training, ablation and statistical comparison.

* Every candidate is trained **from scratch** on its own feature schema
  (the base weights are never touched; an environmental input shape cannot
  reuse them anyway). Lineage records the parent (current champion) and
  ``initialisation = scratch``.
* Candidates are compared on **identical samples**: the evaluation datasets
  hold every feature column and each model reads its own columns by name.
  Only samples complete for all compared schemas are used (complete-case).
* The best candidate is chosen on the *validation* period; test / holdout
  periods are only used for the final champion / base comparison, so model
  selection never peeks at the test data.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.constants import ARCHITECTURE_VERSION, FORECAST_DAYS
from ml.environment.alignment import complete_mask
from ml.evaluation.evaluator import EvaluationResult, evaluate_bundle, prediction_errors
from ml.features.schemas import FeatureSchema
from ml.models.gru_architecture import ENV_CONCAT_ARCHITECTURE_VERSION, build_model
from ml.models.model_loader import ModelBundle
from ml.training.dataset_builder import SequenceDataset
from ml.training.trainer import TrainingConfig, fit, fit_scalers, transform


@dataclass
class CandidateOutcome:
    schema: FeatureSchema
    bundle: ModelBundle
    architecture_version: str
    history: dict[str, list[float]]
    train_samples: int
    validation_samples: int
    validation: EvaluationResult
    started_at: float
    completed_at: float
    initialisation: str = "scratch"


def complete_subset(ds: SequenceDataset, features: Sequence[str]) -> SequenceDataset:
    return ds.subset(complete_mask(ds, features))


def architecture_for(schema: FeatureSchema) -> str:
    return ENV_CONCAT_ARCHITECTURE_VERSION if schema.is_environmental else ARCHITECTURE_VERSION


def train_schema_candidate(
    schema: FeatureSchema,
    train: SequenceDataset,
    validation: SequenceDataset,
    config: TrainingConfig,
    version: str,
    min_samples: int = 50,
) -> CandidateOutcome:
    features = schema.features
    tr = complete_subset(train, features)
    va = complete_subset(validation, features)
    if len(tr) < min_samples or len(va) == 0:
        raise ValueError(
            f"{schema.version}: not enough complete samples (train {len(tr)}, validation {len(va)}; need {min_samples}+ / 1+)"
        )
    started = time.time()
    arch = architecture_for(schema)
    X_tr = tr.select(features)
    fs, ts = fit_scalers(X_tr, tr.y)
    Xtr, ytr = transform(fs, ts, X_tr, tr.y)
    Xva, yva = transform(fs, ts, va.select(features), va.y)
    model = build_model(arch, n_features=len(features))
    history = fit(model, Xtr, ytr, Xva, yva, config)
    bundle = ModelBundle(
        version=version, model=model, feature_scaler=fs, target_scaler=ts,
        metadata={"architecture_version": arch}, directory=Path("."),
        feature_names=features, feature_schema_version=schema.version,
    )
    return CandidateOutcome(
        schema=schema, bundle=bundle, architecture_version=arch, history=history,
        train_samples=len(tr), validation_samples=len(va),
        validation=evaluate_bundle(bundle, va, "validation"),
        started_at=started, completed_at=time.time(),
    )


def select_best(outcomes: Sequence[CandidateOutcome], horizon: int = 7) -> CandidateOutcome:
    """Lowest validation MAE at ``horizon`` (validation period only)."""
    scored = [(o.validation.by_horizon[horizon].mae_km, o) for o in outcomes if o.validation.by_horizon[horizon].mae_km is not None]
    if not scored:
        raise ValueError("no candidate has validation labels at the primary horizon")
    return min(scored, key=lambda t: t[0])[1]


# ---------------------------------------------------------------- statistics
def paired_comparison(
    reference_errors: np.ndarray, candidate_errors: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> dict[str, dict[str, float | int | None]]:
    """Per horizon, on identical samples: MAE difference (candidate − reference)
    with a paired bootstrap 95 % CI, and the share of samples where the
    candidate error is smaller. Negative difference = candidate better."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float | int | None]] = {}
    for h in range(FORECAST_DAYS):
        r, c = reference_errors[:, h], candidate_errors[:, h]
        ok = np.isfinite(r) & np.isfinite(c)
        n = int(ok.sum())
        if n == 0:
            out[str(h + 1)] = {"n": 0, "mae_diff_km": None, "ci95_low_km": None, "ci95_high_km": None, "candidate_better_share": None}
            continue
        d = c[ok] - r[ok]
        boots = [float(d[rng.integers(0, n, n)].mean()) for _ in range(n_boot)] if n > 1 else [float(d.mean())]
        out[str(h + 1)] = {
            "n": n,
            "mae_diff_km": float(d.mean()),
            "ci95_low_km": float(np.quantile(boots, 0.025)),
            "ci95_high_km": float(np.quantile(boots, 0.975)),
            "candidate_better_share": float((c[ok] < r[ok]).mean()),
        }
    return out


def error_distribution(errors: np.ndarray) -> dict[str, dict[str, float | None]]:
    qs = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    out: dict[str, dict[str, float | None]] = {}
    for h in range(FORECAST_DAYS):
        e = errors[:, h]
        e = e[np.isfinite(e)]
        out[str(h + 1)] = {f"p{int(q * 100)}": (float(np.quantile(e, q)) if e.size else None) for q in qs}
    return out


def ablation_report(
    bundles: Mapping[str, ModelBundle], dataset: SequenceDataset, protocol: str, reference: str = "base"
) -> dict[str, Any]:
    """Score every model on the same samples; answer "did X improve the forecast?".

    ``bundles`` maps a label (``base``, ``champion:v1``, schema versions …) to a
    model. Returns per-model metrics, error distributions, paired comparisons
    against ``reference`` and per-variable-group effects where the schema pair
    exists (e.g. wind effect = traj_wind_v1 vs trajectory_v1).
    """
    errors = {label: prediction_errors(b, dataset) for label, b in bundles.items()}
    from ml.evaluation.evaluator import result_from_errors

    results = {label: result_from_errors(protocol, e) for label, e in errors.items()}
    report: dict[str, Any] = {
        "protocol": protocol,
        "samples": len(dataset),
        "anchor_date_range": list(dataset.date_range()),
        "models": {label: {"schema": bundles[label].feature_schema_version, **r.as_dict()} for label, r in results.items()},
        "distributions": {label: error_distribution(e) for label, e in errors.items()},
        "vs_reference": {},
        "effects": {},
    }
    if reference in errors:
        report["vs_reference"] = {
            label: paired_comparison(errors[reference], e) for label, e in errors.items() if label != reference
        }
    by_schema = {b.feature_schema_version: label for label, b in bundles.items() if label != reference}
    pairs = {
        "wind": ("trajectory_v1", "traj_wind_v1"),
        "ocean_current": ("trajectory_v1", "traj_current_v1"),
        "wind_plus_current": ("trajectory_v1", "traj_wind_current_v1"),
        "sea_ice_given_wind_current": ("traj_wind_current_v1", "traj_wind_current_ice_v1"),
    }
    for effect, (without, with_) in pairs.items():
        a = by_schema.get(without) or (reference if without == "trajectory_v1" and reference in errors else None)
        b = by_schema.get(with_)
        if a and b:
            cmp = paired_comparison(errors[a], errors[b])
            report["effects"][effect] = {
                "without": a,
                "with": b,
                "by_horizon": cmp,
                "verdict": {
                    h: ("improved" if v["ci95_high_km"] is not None and v["ci95_high_km"] < 0
                        else "worse" if v["ci95_low_km"] is not None and v["ci95_low_km"] > 0
                        else "no significant difference" if v["n"] else "no data")
                    for h, v in cmp.items()
                },
            }
    return report
