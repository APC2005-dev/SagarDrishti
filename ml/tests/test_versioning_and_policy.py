from __future__ import annotations

from pathlib import Path

import pytest

from ml.evaluation.champion_challenger import PromotionPolicy, decide
from ml.evaluation.evaluator import EvaluationResult
from ml.evaluation.metrics import ErrorSummary, headline_errors, summarize, summarize_by_horizon
from ml.versioning.version_manager import format_version, next_version, version_number


class TestVersions:
    def test_unbounded_numbering(self) -> None:
        assert next_version([]) == "v1"
        assert next_version(["base", "v1", "v2", "v9"]) == "v10"
        assert next_version([f"v{i}" for i in range(1, 250)]) == "v250"

    def test_filesystem_orphans_are_not_reused(self, tmp_path: Path) -> None:
        (tmp_path / "v7").mkdir()
        assert next_version(["v1", "v2"], tmp_path) == "v8"

    def test_parse(self) -> None:
        assert version_number("base") == 0
        assert version_number("v42") == 42
        for bad in ("v0", "V3", "v-1", "3", "v01", "latest"):
            with pytest.raises(ValueError):
                version_number(bad)
        with pytest.raises(ValueError):
            format_version(0)


class TestMetrics:
    def test_summary(self) -> None:
        s = summarize([1.0, 2.0, 3.0, 4.0, float("nan")])
        assert s.n == 4
        assert s.mae_km == pytest.approx(2.5)
        assert s.rmse_km == pytest.approx((30 / 4) ** 0.5)
        assert s.median_km == pytest.approx(2.5)
        assert s.p90_km == pytest.approx(3.7)

    def test_empty(self) -> None:
        assert summarize([]).mae_km is None

    def test_by_horizon(self) -> None:
        by = summarize_by_horizon([(1, 1.0), (7, 10.0), (7, 20.0)])
        assert headline_errors(by) == {"day1_error": 1.0, "day3_error": None, "day7_error": 15.0}


def _result(protocol: str, maes: dict[int, float], n: int = 100) -> EvaluationResult:
    by = {h: ErrorSummary(n, m, m, m, m) for h, m in maes.items()}
    return EvaluationResult(protocol, by, ErrorSummary(n, 1, 1, 1, 1))


class TestChampionChallenger:
    policy = PromotionPolicy(min_relative_improvement=0.02, max_short_horizon_regression=0.05, min_evaluation_samples=30)

    def test_promotes_clear_improvement(self) -> None:
        d = decide(_result("p", {1: 1.0, 3: 3.0, 7: 10.0}), _result("p", {1: 1.0, 3: 3.0, 7: 9.0}), self.policy)
        assert d.promote, d.reasons

    def test_rejects_worse_day7(self) -> None:
        d = decide(_result("p", {1: 1.0, 3: 3.0, 7: 10.0}), _result("p", {1: 0.5, 3: 2.0, 7: 10.5}), self.policy)
        assert not d.promote

    def test_rejects_marginal_improvement_below_threshold(self) -> None:
        d = decide(_result("p", {7: 10.0}), _result("p", {7: 9.9}), self.policy)
        assert not d.promote

    def test_rejects_short_horizon_regression(self) -> None:
        d = decide(_result("p", {1: 1.0, 3: 3.0, 7: 10.0}), _result("p", {1: 1.2, 3: 3.0, 7: 8.0}), self.policy)
        assert not d.promote
        assert any("day-1 regression" in r for r in d.reasons)

    def test_rejects_too_few_samples(self) -> None:
        d = decide(_result("p", {7: 10.0}, n=5), _result("p", {7: 5.0}, n=5), self.policy)
        assert not d.promote

    def test_protocol_mismatch(self) -> None:
        with pytest.raises(ValueError):
            decide(_result("a", {7: 1.0}), _result("b", {7: 1.0}), self.policy)

    def test_missing_primary_labels(self) -> None:
        d = decide(_result("p", {1: 1.0}), _result("p", {1: 0.5}), self.policy)
        assert not d.promote
