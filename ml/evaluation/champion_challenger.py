"""Champion / challenger promotion policy.

Both models are evaluated on *identical* held-out data (same protocol, same
samples) before this is called. A challenger is promoted only if every rule
passes; a successful training run alone never changes the champion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ml.evaluation.evaluator import EvaluationResult


@dataclass(frozen=True)
class PromotionPolicy:
    primary_horizon: int = 7
    # Challenger primary MAE must be <= champion * (1 - min_relative_improvement).
    min_relative_improvement: float = 0.02
    # No horizon in guard_horizons may worsen by more than this fraction.
    max_short_horizon_regression: float = 0.05
    guard_horizons: tuple[int, ...] = (1, 3)
    # Minimum labelled samples at the primary horizon for the comparison to count.
    min_evaluation_samples: int = 30

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    promote: bool
    protocol: str
    reasons: list[str] = field(default_factory=list)
    comparisons: dict[str, dict[str, float | int | None]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(
    champion: EvaluationResult, challenger: EvaluationResult, policy: PromotionPolicy
) -> PromotionDecision:
    if champion.protocol != challenger.protocol:
        raise ValueError("champion and challenger must be evaluated on the same protocol")
    decision = PromotionDecision(promote=True, protocol=challenger.protocol)

    horizons = sorted({policy.primary_horizon, *policy.guard_horizons})
    for h in horizons:
        c, n = champion.by_horizon.get(h), challenger.by_horizon.get(h)
        decision.comparisons[str(h)] = {
            "champion_mae_km": c.mae_km if c else None,
            "challenger_mae_km": n.mae_km if n else None,
            "n": n.n if n else 0,
        }

    primary_c = champion.by_horizon.get(policy.primary_horizon)
    primary_n = challenger.by_horizon.get(policy.primary_horizon)
    if not primary_c or not primary_n or primary_c.mae_km is None or primary_n.mae_km is None:
        decision.promote = False
        decision.reasons.append(f"no day-{policy.primary_horizon} labels on protocol {challenger.protocol}")
        return decision
    if primary_n.n < policy.min_evaluation_samples:
        decision.promote = False
        decision.reasons.append(
            f"only {primary_n.n} day-{policy.primary_horizon} samples (< {policy.min_evaluation_samples})"
        )

    threshold = primary_c.mae_km * (1 - policy.min_relative_improvement)
    if primary_n.mae_km <= threshold:
        decision.reasons.append(
            f"day-{policy.primary_horizon} MAE {primary_n.mae_km:.3f} km <= required {threshold:.3f} km "
            f"(champion {primary_c.mae_km:.3f} km)"
        )
    else:
        decision.promote = False
        decision.reasons.append(
            f"day-{policy.primary_horizon} MAE {primary_n.mae_km:.3f} km does not beat required {threshold:.3f} km "
            f"(champion {primary_c.mae_km:.3f} km)"
        )

    for h in policy.guard_horizons:
        c, n = champion.by_horizon.get(h), challenger.by_horizon.get(h)
        if not c or not n or c.mae_km is None or n.mae_km is None or c.n == 0:
            continue
        limit = c.mae_km * (1 + policy.max_short_horizon_regression)
        if n.mae_km > limit:
            decision.promote = False
            decision.reasons.append(f"day-{h} regression: {n.mae_km:.3f} km > limit {limit:.3f} km")
    return decision
