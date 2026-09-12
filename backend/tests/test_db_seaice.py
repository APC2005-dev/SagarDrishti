"""Sea-ice model: ingestion, the 7-entry window, versioning and champion/challenger.

Covers TEST 1-10 of the sea-ice specification. The Copernicus source is faked so
nothing here reaches the network, but the grids have the real shape and go
through the real preprocessing, loader, registry and artifact store.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.models.ml import ModelVersion
from app.models.seaice import SeaIceObservation
from app.services.model_registry import ModelRegistry, RegistryError
from app.services.seaice_ingestion_service import (
    SeaIceIngestionService,
    load_recent_entries,
    stack_entries,
)
from app.services.seaice_registry import SeaIceRegistry
from ml.seaice.constants import GRID_SHAPE, HORIZONS, IN_CHANNELS, OUT_CHANNELS, WINDOW
from ml.seaice.dataset import build_samples, chronological_split
from ml.seaice.preprocessing import build_input_tensor, first_target_date
from ml.versioning.version_manager import qualify
from tests.helpers import write_tiny_seaice_base

pytestmark = [pytest.mark.db, pytest.mark.tf]

START = date(2026, 6, 1)


class FakeDataArray:
    """Minimal stand-in for the xarray object ``coarsen_ice_conc`` receives."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __truediv__(self, divisor: float) -> FakeDataArray:
        return FakeDataArray(self.values / divisor)

    def coarsen(self, **_: object) -> FakeDataArray:
        return self

    def mean(self, **_: object) -> FakeDataArray:
        return self

    def load(self) -> FakeDataArray:
        return self


class FakeDataset:
    """A fake OSI SAF dataset already on the coarsened grid, with controllable dates."""

    def __init__(self, days: list[date]) -> None:
        self.days = days
        self._time = FakeDataArray(np.array([np.datetime64(d.isoformat()) for d in days]))
        rng = np.random.default_rng(7)
        self._field = rng.uniform(0, 100, size=(len(days), *GRID_SHAPE)).astype(np.float32)

    def __getitem__(self, key: str) -> object:
        if key == "time":
            return self._time
        return _FakeVariable(self._field)


class _FakeVariable:
    """Supports both the chunked read (``isel(time=slice)``) and per-step access."""

    def __init__(self, field: np.ndarray) -> None:
        self.field = field

    def isel(self, time: int | slice) -> _FakeVariable | FakeDataArray:
        selected = self.field[time]
        return _FakeVariable(selected) if isinstance(time, slice) else FakeDataArray(selected)

    def load(self) -> _FakeVariable:
        return self


def fake_source(days: list[date]):
    from ml.seaice.source import SeaIceSource

    return SeaIceSource(opener=lambda dataset_id: FakeDataset(days))


def ingest(db, settings, days: list[date]):
    return SeaIceIngestionService(db, settings, source=fake_source(days)).run(trigger="test")


def register_seaice(db, version: str, status: str, *, parent: str | None = None, day7: float | None = None):
    """Insert a sea-ice version row directly, as a completed training run would."""
    mv = ModelVersion(
        version=version, model_family="sea_ice", model_type="sea_ice",
        version_number=int(version.rpartition("/")[2][1:]), parent_version=parent,
        feature_schema_version="unet_residual_v4_entry7", architecture="UNetResidual",
        architecture_version="unet_residual_v4", input_sequence_length=WINDOW,
        input_semantics="chronological_observation_entries", forecast_horizon_days=max(HORIZONS),
        feature_names=[], adapter_strategy=None, model_path=f"/tmp/{version}/model.pt",
        scaler_path="", metadata_path=f"/tmp/{version}/metadata.json", artifact_sha256={},
        artifact_origin="test", status=status, day7_error=day7,
    )
    db.add(mv)
    db.flush()
    return mv


# ------------------------------------------------------------------- TEST 1
def test_case1_base_model_loads(db, settings) -> None:  # type: ignore[no-untyped-def]
    write_tiny_seaice_base(Path(settings.models_dir))
    registry = SeaIceRegistry(db, settings)
    base = registry.register_base()
    assert base.version == qualify("sea_ice", "base") and base.model_family == "sea_ice"
    bundle = registry.load_bundle(base)
    assert sum(p.numel() for p in bundle.model.parameters()) == 483_939
    import torch

    with torch.no_grad():
        out = bundle.model(torch.zeros(1, IN_CHANNELS, *GRID_SHAPE))
    assert tuple(out.shape) == (1, OUT_CHANNELS, *GRID_SHAPE)


# ------------------------------------------------------------- TEST 2 and 3
def test_case2_and_3_ingestion_is_idempotent_and_keeps_history(db, settings) -> None:  # type: ignore[no-untyped-def]
    days = [START + timedelta(days=i) for i in range(10)]
    first = ingest(db, settings, days)
    assert first.status == "success" and first.new == 10

    # TEST 2: nothing new at the source -> no duplicate row, no new work.
    again = ingest(db, settings, days)
    assert again.status == "unchanged" and again.new == 0
    assert db.query(SeaIceObservation).count() == 10

    # TEST 3: a new official entry is stored once and history stays intact.
    extended = [*days, days[-1] + timedelta(days=1)]
    third = ingest(db, settings, extended)
    assert third.status == "success" and third.new == 1
    stored = sorted(o.observation_date for o in db.query(SeaIceObservation).all())
    assert len(stored) == 11 and stored[0] == days[0]  # oldest entry not deleted


def test_official_observations_are_append_only(db, settings) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import text

    ingest(db, settings, [START + timedelta(days=i) for i in range(8)])
    db.flush()
    with pytest.raises(Exception, match="append-only"):
        db.execute(text("UPDATE seaice.observations SET mean_concentration = 0.5"))
    db.rollback()


# ------------------------------------------------------------------- TEST 4
def test_case4_window_is_seven_entries_not_seven_calendar_days(db, settings) -> None:  # type: ignore[no-untyped-def]
    """The source has gaps; the window is still the last 7 STORED entries."""
    days = [START, START + timedelta(days=1), START + timedelta(days=2),
            START + timedelta(days=9),   # gap
            START + timedelta(days=10), START + timedelta(days=11),
            START + timedelta(days=25),  # bigger gap
            START + timedelta(days=26)]
    ingest(db, settings, days)
    entries = load_recent_entries(db, WINDOW)
    assert len(entries) == WINDOW
    assert [e.observation_date for e in entries] == days[-WINDOW:]  # 7 entries, ascending
    span = (entries[-1].observation_date - entries[0].observation_date).days
    assert span == 25 and span != WINDOW - 1  # 7 entries spanning 25 calendar days
    concentration, mask = stack_entries(entries)
    tensor = build_input_tensor(concentration, mask, first_target_date(entries[-1].observation_date))
    assert tensor.shape == (IN_CHANNELS, *GRID_SHAPE)
    # No day was fabricated to pad the window.
    assert db.query(SeaIceObservation).count() == len(days)


def test_samples_skip_missing_targets_instead_of_interpolating(db, settings) -> None:  # type: ignore[no-untyped-def]
    days = [START + timedelta(days=i) for i in range(20) if i != 15]  # 2026-06-16 missing
    ingest(db, settings, days)
    entries = list(db.query(SeaIceObservation).order_by(SeaIceObservation.observation_date).all())
    concentration, mask = stack_entries(entries)
    samples, report = build_samples([e.observation_date for e in entries], concentration, mask)
    assert report.skipped_missing_target > 0
    assert all(s.entry_dates[-1] + timedelta(days=h) in set(days) for s in samples for h in HORIZONS)


# ------------------------------------------------------------- TEST 5 and 9
def test_case5_next_version_is_allocated_dynamically(db, settings) -> None:  # type: ignore[no-untyped-def]
    """No version is manufactured: the base IS the first champion, and the first
    numbered version is allocated only when retraining produces one."""
    write_tiny_seaice_base(Path(settings.models_dir))
    registry = SeaIceRegistry(db, settings)
    registry.register_base()
    first = registry.ensure_champion()
    assert first.version == qualify("sea_ice", "base") and first.version_number == 0
    assert registry.allocate_next_version() == qualify("sea_ice", "v1")  # derived, not a literal
    for n in range(1, 7):
        register_seaice(db, qualify("sea_ice", f"v{n}"), "rejected",
                        parent=qualify("sea_ice", f"v{n-1}") if n > 1 else None)
    assert registry.allocate_next_version() == qualify("sea_ice", "v7")


def test_case9_champion_is_not_the_highest_version(db, settings) -> None:  # type: ignore[no-untyped-def]
    """v1..v5 exist and v4 is the champion: the application must load v4."""
    for n in range(1, 6):
        register_seaice(
            db, qualify("sea_ice", f"v{n}"),
            "deployed" if n == 4 else ("rejected" if n == 5 else "archived"),
            parent=qualify("sea_ice", f"v{n-1}") if n > 1 else None,
        )
    db.commit()
    registry = SeaIceRegistry(db, settings)
    assert registry.latest_version().version == qualify("sea_ice", "v5")
    assert registry.get_champion().version == qualify("sea_ice", "v4")
    assert registry.ensure_champion().version == qualify("sea_ice", "v4")  # never resurrects v1


# ------------------------------------------------------------- TEST 6 and 7
def test_case6_and_7_promotion_and_rejection(db, settings) -> None:  # type: ignore[no-untyped-def]
    registry = SeaIceRegistry(db, settings)
    champion = register_seaice(db, qualify("sea_ice", "v1"), "deployed", day7=0.09)

    # TEST 7: a worse candidate is not promoted and the champion is untouched.
    worse = register_seaice(db, qualify("sea_ice", "v2"), "validated", parent=champion.version, day7=0.12)
    registry.transition(worse, "rejected", "did not beat the champion at D+7")
    assert registry.get_champion().version == qualify("sea_ice", "v1")
    assert registry.get(qualify("sea_ice", "v2")).status == "rejected"  # kept for lineage

    # TEST 6: a better candidate becomes champion; the old one is archived, not deleted.
    better = register_seaice(db, qualify("sea_ice", "v3"), "validated", parent=champion.version, day7=0.07)
    registry.promote(better, "beat the champion at D+7")
    assert registry.get_champion().version == qualify("sea_ice", "v3")
    assert registry.get(qualify("sea_ice", "v1")).status == "archived"
    assert {m.version for m in registry.list_versions()} == {
        qualify("sea_ice", v) for v in ("v1", "v2", "v3")
    }


def test_rejected_version_cannot_be_deployed_but_archived_can_roll_back(db, settings) -> None:  # type: ignore[no-untyped-def]
    registry = SeaIceRegistry(db, settings)
    register_seaice(db, qualify("sea_ice", "v1"), "archived")
    rejected = register_seaice(db, qualify("sea_ice", "v2"), "rejected")
    register_seaice(db, qualify("sea_ice", "v3"), "deployed")
    registry.promote(registry.get(qualify("sea_ice", "v1")), "rollback: v3 regressed")
    assert registry.get_champion().version == qualify("sea_ice", "v1")
    with pytest.raises(RegistryError):
        registry.transition(rejected, "deployed", "should be impossible")


# ------------------------------------------------------------------- TEST 8
def test_case8_champion_survives_restart(db, settings) -> None:  # type: ignore[no-untyped-def]
    register_seaice(db, qualify("sea_ice", "v1"), "archived")
    register_seaice(db, qualify("sea_ice", "v2"), "archived", parent=qualify("sea_ice", "v1"))
    register_seaice(db, qualify("sea_ice", "v3"), "deployed", parent=qualify("sea_ice", "v2"))
    db.commit()
    db.expunge_all()  # a restart keeps nothing in memory
    assert SeaIceRegistry(db, settings).get_champion().version == qualify("sea_ice", "v3")


# ------------------------------------------------------------------ TEST 10
def test_case10_families_are_independent(db, settings) -> None:  # type: ignore[no-untyped-def]
    """Trajectory v4 and sea-ice v4 are unrelated models that share only a number."""
    from tests.helpers import write_tiny_base

    write_tiny_base(Path(settings.models_dir))
    trajectory = ModelRegistry(db, settings).ensure_v1_bootstrap()
    for n in range(2, 5):
        register_seaice(db, qualify("sea_ice", f"v{n}"), "archived" if n < 4 else "deployed")
    register_seaice(db, qualify("sea_ice", "v1"), "archived")
    db.commit()

    seaice = SeaIceRegistry(db, settings)
    assert trajectory.version == "v1" and trajectory.model_family == "trajectory"
    assert seaice.get_champion().version == qualify("sea_ice", "v4")
    # Both families have a champion deployed at the same time.
    assert ModelRegistry(db, settings).get_deployed().version == "v1"
    # Neither registry sees the other's versions.
    assert all(m.model_family == "trajectory" for m in ModelRegistry(db, settings).list_versions())
    assert all(m.model_family == "sea_ice" for m in seaice.list_versions())
    # Same number, different models.
    assert seaice.get_champion().version_number == 4
    assert seaice.get_champion().architecture == "UNetResidual"
    assert ModelRegistry(db, settings).get_deployed().architecture == "GRU"


def test_seaice_version_numbers_do_not_collide_with_trajectory(db, settings) -> None:  # type: ignore[no-untyped-def]
    from tests.helpers import write_tiny_base

    write_tiny_base(Path(settings.models_dir))
    ModelRegistry(db, settings).ensure_v1_bootstrap()   # trajectory v1, version_number 1
    register_seaice(db, qualify("sea_ice", "v1"), "deployed")  # sea-ice v1, version_number 1
    db.flush()
    numbers = {(m.model_family, m.version_number) for m in db.query(ModelVersion).all()}
    assert ("trajectory", 1) in numbers and ("sea_ice", 1) in numbers


# ------------------------------------------------------------ forecast path
def test_forecast_uses_champion_and_seven_entries(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_forecast_service import SeaIceForecastService

    write_tiny_seaice_base(Path(settings.models_dir))
    registry = SeaIceRegistry(db, settings)
    registry.register_base()
    champion = registry.ensure_champion()
    ingest(db, settings, [START + timedelta(days=i) for i in range(8)])

    out = SeaIceForecastService(db, settings).run(trigger="test")
    assert out.status == "success" and out.created == len(HORIZONS)
    assert out.model_version == champion.version
    assert len(out.diagnostics["input_entry_dates"]) == WINDOW

    # Idempotent for the same anchor and model.
    assert SeaIceForecastService(db, settings).run(trigger="test").status == "unchanged"


def test_forecast_skips_rather_than_padding_a_short_window(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_forecast_service import SeaIceForecastService

    write_tiny_seaice_base(Path(settings.models_dir))
    SeaIceRegistry(db, settings).register_base()
    SeaIceRegistry(db, settings).ensure_champion()
    ingest(db, settings, [START + timedelta(days=i) for i in range(4)])  # only 4 entries
    out = SeaIceForecastService(db, settings).run(trigger="test")
    assert out.status == "skipped" and "4 sea-ice entries" in out.error


def test_evaluation_scores_against_official_data_only(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_evaluation_service import SeaIceEvaluationService
    from app.services.seaice_forecast_service import SeaIceForecastService

    write_tiny_seaice_base(Path(settings.models_dir))
    SeaIceRegistry(db, settings).register_base()
    SeaIceRegistry(db, settings).ensure_champion()
    ingest(db, settings, [START + timedelta(days=i) for i in range(8)])
    SeaIceForecastService(db, settings).run(trigger="test")

    # Targets are in the future: nothing to score yet.
    pending = SeaIceEvaluationService(db, settings).run(trigger="test")
    assert pending.evaluated == 0 and pending.pending == len(HORIZONS)

    # The official entries for the target dates arrive.
    ingest(db, settings, [START + timedelta(days=i) for i in range(16)])
    scored = SeaIceEvaluationService(db, settings).run(trigger="test")
    assert scored.evaluated == len(HORIZONS)
    from app.models.seaice import SeaIceEvaluation

    rows = db.query(SeaIceEvaluation).all()
    assert all(r.rmse >= 0 and r.persistence_rmse is not None for r in rows)


def test_chronological_split_never_shuffles(db, settings) -> None:  # type: ignore[no-untyped-def]
    ingest(db, settings, [START + timedelta(days=i) for i in range(24)])
    entries = list(db.query(SeaIceObservation).order_by(SeaIceObservation.observation_date).all())
    concentration, mask = stack_entries(entries)
    samples, _ = build_samples([e.observation_date for e in entries], concentration, mask)
    train, validation = chronological_split(samples, 0.25)
    assert train and validation
    assert max(s.anchor_date for s in train) <= min(s.anchor_date for s in validation)


# ------------------------------------------- retraining, end to end (TEST 5-7)
def _seed_for_retraining(db, settings):
    write_tiny_seaice_base(Path(settings.models_dir))
    registry = SeaIceRegistry(db, settings)
    registry.register_base()
    champion = registry.ensure_champion()
    ingest(db, settings, [START + timedelta(days=i) for i in range(26)])
    return registry, champion


def _fast(settings, **overrides):
    """Tiny training budget; thresholds chosen to force a deterministic decision."""
    return settings.model_copy(update={
        "seaice_retrain_epochs": 1, "seaice_retrain_batch_size": 4, "seaice_retrain_patience": 1,
        "seaice_retrain_validation_fraction": 0.3, "seaice_promotion_min_evaluation_samples": 1,
        **overrides,
    })


def test_retraining_creates_a_dynamically_numbered_candidate_and_promotes(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_retraining_service import SeaIceRetrainingService

    registry, champion = _seed_for_retraining(db, settings)
    # min improvement -1.0 => any candidate clears the bar, so promotion is exercised.
    fast = _fast(settings, seaice_promotion_min_relative_improvement=-1.0)
    out = SeaIceRetrainingService(db, fast).run(trigger="test", force=True)

    assert out.status == "success", out.error
    assert out.candidate_version == qualify("sea_ice", "v1")  # first TRAINED version, allocated
    assert out.decision == "promoted"
    assert registry.get_champion().version == qualify("sea_ice", "v1")
    assert registry.get(champion.version).status == "archived"  # base kept, not deleted
    # Champion and candidate were scored on the same samples.
    assert out.comparison["champion"]["samples"] == out.comparison["candidate"]["samples"]
    assert (Path(settings.models_dir) / "sea_ice" / "v1" / "model.pt").exists()


def test_retraining_keeps_champion_when_candidate_is_not_better(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_retraining_service import SeaIceRetrainingService

    registry, champion = _seed_for_retraining(db, settings)
    # An impossible bar => the candidate is rejected but still registered.
    fast = _fast(settings, seaice_promotion_min_relative_improvement=0.99)
    out = SeaIceRetrainingService(db, fast).run(trigger="test", force=True)

    assert out.status == "success" and out.decision == "rejected"
    assert registry.get_champion().version == champion.version  # untouched
    assert registry.get(out.candidate_version).status == "rejected"  # preserved for lineage
    assert (Path(settings.models_dir) / "sea_ice" / "v1" / "model.pt").exists()


def test_retraining_is_not_triggered_merely_because_the_scheduler_ran(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_retraining_service import SeaIceRetrainingService

    _seed_for_retraining(db, settings)
    out = SeaIceRetrainingService(db, settings).run(trigger="scheduled")  # not forced
    assert out.status == "skipped" and out.decision == "not eligible"
    assert out.candidate_version is None
    assert not (Path(settings.models_dir) / "sea_ice" / "v1").exists()  # no version created
    checks = out.eligibility["checks"]
    assert "performance_degradation" in checks  # the configured threshold gates it


def test_base_artifact_is_never_written_to(db, settings) -> None:  # type: ignore[no-untyped-def]
    from app.services.seaice_retraining_service import SeaIceRetrainingService
    from ml.adapters.bootstrap_adapter import sha256_file
    from ml.seaice.constants import BASE_ARTIFACT_FILENAME

    registry, _ = _seed_for_retraining(db, settings)
    base_file = Path(settings.models_dir) / "sea_ice" / "base" / BASE_ARTIFACT_FILENAME
    before = sha256_file(base_file)
    SeaIceRetrainingService(db, _fast(settings, seaice_promotion_min_relative_improvement=-1.0)).run(
        trigger="test", force=True
    )
    assert sha256_file(base_file) == before  # byte-identical after a full retraining run
    base = registry.get(qualify("sea_ice", "base"))
    # The base may be superseded as champion, but its artifact is never rewritten
    # and the row is never deleted.
    assert base is not None and base.artifact_origin == "test_fixture"
    assert base_file.exists()


# --------------------------------------------------- historical bulk backfill
def test_backfill_loads_history_beyond_the_poll_cap(db, settings) -> None:  # type: ignore[no-untyped-def]
    """``load-historical`` is a bulk catch-up, not limited by the per-poll cap."""
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(60)]
    capped = settings.model_copy(update={"seaice_max_entries_per_run": 5})
    service = SeaIceIngestionService(db, capped, source=fake_source(days))

    poll = service.run(trigger="scheduled")
    assert poll.new == 5  # the poller respects its cap

    history = service.backfill(since=days[0])
    assert history.new == 55 and history.status == "success"
    assert db.query(SeaIceObservation).count() == 60


def test_backfill_respects_the_configured_start_and_is_rerunnable(db, settings) -> None:  # type: ignore[no-untyped-def]
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(40)]
    cutoff = date(2026, 1, 21)
    tuned = settings.model_copy(update={"seaice_historical_start": cutoff})
    service = SeaIceIngestionService(db, tuned, source=fake_source(days))

    first = service.backfill()
    assert first.new == 20
    stored = sorted(o.observation_date for o in db.query(SeaIceObservation).all())
    assert stored[0] == cutoff  # nothing before the configured start

    # Re-running stores nothing again and deletes nothing.
    again = service.backfill()
    assert again.new == 0 and again.status == "unchanged"
    assert db.query(SeaIceObservation).count() == 20


def test_backfill_fetches_contiguous_days_in_chunks(db, settings) -> None:  # type: ignore[no-untyped-def]
    """Gaps split a chunk, so a read never drags in days that were not requested."""
    from ml.seaice.source import _contiguous_groups

    pairs = [(0, date(2026, 1, 1)), (1, date(2026, 1, 2)), (2, date(2026, 1, 3)),
             (7, date(2026, 1, 8)), (8, date(2026, 1, 9))]
    groups = [[index for index, _ in group] for group in _contiguous_groups(pairs, chunk_size=10)]
    assert groups == [[0, 1, 2], [7, 8]]
    assert [[i for i, _ in g] for g in _contiguous_groups(pairs, chunk_size=2)] == [[0, 1], [2], [7, 8]]


def test_load_historical_cli_loads_both_families(db, settings, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``load-historical`` covers icebergs AND sea ice in one command."""
    from app import cli

    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(12)]
    monkeypatch.setattr(
        cli, "_seaice_history",
        lambda session, s, since, until: SeaIceIngestionService(
            db, s, source=fake_source(days)
        ).backfill(since=days[0]).as_dict(),
    )
    result = cli._seaice_history(db, settings, None, None)
    assert result["entries_new"] == 12
    assert db.query(SeaIceObservation).count() == 12
