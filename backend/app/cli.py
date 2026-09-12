"""Operator CLI (runs jobs synchronously, no broker required).

    python -m app.cli load-historical            # BYU icebergs + official sea-ice history
    python -m app.cli seaice-load-historical     # sea-ice history only
    python -m app.cli register-base              # register models/base
    python -m app.cli bootstrap                  # both core families: trajectory + sea ice
    python -m app.cli ingest [--file x.csv]      # USNIC icebergs + Copernicus sea ice
    python -m app.cli evaluate                   # score forecasts vs official observations
    python -m app.cli forecast                   # trajectory + sea-ice champions
    python -m app.cli retrain [--force]          # policy-gated retraining
    python -m app.cli benchmark --version v1     # all-horizon historical benchmark
    python -m app.cli status
    python -m app.cli seaice-register-base   # register the supplied U-Net Residual v4 .pt
    python -m app.cli seaice-bootstrap       # sea-ice family only
    python -m app.cli seaice-ingest          # poll Copernicus Marine for new official fields
    python -m app.cli seaice-forecast        # forecast from the sea-ice champion
    python -m app.cli seaice-evaluate        # score sea-ice forecasts vs official data
    python -m app.cli seaice-retrain [--force]
    python -m app.cli seaice-status
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import session_scope
from ml.models.model_loader import ModelArtifactError


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level, json=False)
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_lh = sub.add_parser("load-historical")
    p_lh.add_argument("--skip-seaice", action="store_true", help="Load only the BYU iceberg dataset")
    p_lh.add_argument("--seaice-since", type=date.fromisoformat, default=None,
                      help="Override SEAICE_HISTORICAL_START (YYYY-MM-DD)")
    sub.add_parser("register-base")
    p_bs = sub.add_parser("bootstrap", aliases=["bootstrap-v1"])
    p_bs.add_argument("--family", choices=["trajectory", "sea_ice"], default=None,
                      help="Only this model family (default: both core families)")
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--skip-seaice", action="store_true", help="USNIC only")
    p_ing.add_argument("--file", type=Path, help="Import a local USNIC CSV instead of fetching")
    sub.add_parser("evaluate")
    p_fc = sub.add_parser("forecast")
    p_fc.add_argument("--skip-seaice", action="store_true", help="Trajectory only")
    p_rt = sub.add_parser("retrain")
    p_rt.add_argument("--force", action="store_true", help="Ignore eligibility thresholds (promotion policy still applies)")
    p_bm = sub.add_parser("benchmark")
    p_bm.add_argument("--version", required=True)
    sub.add_parser("status")
    sub.add_parser("env-status", help="environmental sources, credentials configured (yes/no), schemas trainable")
    p_ea = sub.add_parser("env-align", help="align environmental state to official observations")
    p_ea.add_argument("--all-latest", action="store_true", help="latest official fix of every active iceberg (default)")
    sub.add_parser("env-overlay", help="refresh coarse overlay grids for the map")
    sub.add_parser("seaice-register-base", help="register the supplied U-Net Residual v4 base model")
    sub.add_parser("seaice-bootstrap", help="create + deploy the first sea-ice production version")
    sub.add_parser("seaice-ingest", help="poll Copernicus Marine for new official sea-ice fields")
    p_slh = sub.add_parser("seaice-load-historical", help="bulk-load official sea-ice history")
    p_slh.add_argument("--since", type=date.fromisoformat, default=None)
    p_slh.add_argument("--until", type=date.fromisoformat, default=None)
    sub.add_parser("seaice-forecast")
    sub.add_parser("seaice-evaluate")
    p_sr = sub.add_parser("seaice-retrain")
    p_sr.add_argument("--force", action="store_true", help="Ignore eligibility (promotion policy still applies)")
    sub.add_parser("seaice-status")
    p_ep = sub.add_parser("env-prefetch", help="pre-warm the environmental cache for a retraining experiment")
    p_ep.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        return _run(args, settings)
    except (ModelArtifactError, FileNotFoundError) as exc:
        # Expected operator errors (e.g. the base model files are not in models/base yet): one clear line, no traceback.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _seaice_history(session: Any, settings: Any, since: Any, until: Any) -> dict[str, Any]:
    """Bulk-load official sea-ice history, reporting rather than raising."""
    from app.services.seaice_ingestion_service import SeaIceIngestionService

    service = SeaIceIngestionService(session, settings)
    ok, reason = service.source.is_configured()
    if not ok:
        return {"status": "skipped", "reason": reason}
    outcome = service.backfill(since=since, until=until)
    summary = outcome.as_dict()
    stored = summary.pop("stored_dates", [])
    summary["first_stored"] = stored[0] if stored else None
    summary["last_stored"] = stored[-1] if stored else None
    summary["total_stored"] = len(service.stored_dates())
    return summary


def _seaice_bootstrap(session: Any, settings: Any) -> dict[str, Any]:
    """Register the supplied U-Net base and make sure the family has a champion.

    No version is manufactured: the base model IS the starting champion, and
    numbered versions appear only when a retraining run produces one.
    """
    if not settings.seaice_enabled:
        return {"status": "skipped", "reason": "SEAICE_ENABLED is false"}
    from app.services.seaice_registry import SeaIceRegistry

    registry = SeaIceRegistry(session, settings)
    registry.register_base()
    mv = registry.ensure_champion()
    return {"version": mv.version, "status": mv.status, "parent": mv.parent_version,
            "architecture_version": mv.architecture_version}


def _run(args: argparse.Namespace, settings: Any) -> int:
    with session_scope() as session:
        if args.cmd == "load-historical":
            from app.services.historical_loader import load_historical_dataset

            result: dict[str, Any] = {"icebergs": load_historical_dataset(session, Path(settings.bootstrap_dataset_path))}
            # The sea-ice model is a core model too, so historical loading covers
            # both families. A sea-ice failure must not lose the iceberg load.
            if args.skip_seaice:
                result["sea_ice"] = {"status": "skipped", "reason": "--skip-seaice"}
            elif not settings.seaice_enabled:
                result["sea_ice"] = {"status": "skipped", "reason": "SEAICE_ENABLED is false"}
            else:
                result["sea_ice"] = _seaice_history(session, settings, args.seaice_since, None)
            _print(result)
        elif args.cmd == "seaice-load-historical":
            _print(_seaice_history(session, settings, args.since, args.until))
        elif args.cmd == "register-base":
            from app.services.model_registry import ModelRegistry

            mv = ModelRegistry(session, settings).register_base()
            _print({"version": mv.version, "status": mv.status, "sha256": mv.artifact_sha256})
        elif args.cmd in ("bootstrap", "bootstrap-v1"):
            # Sagar Drishti has TWO core model families. Bootstrap handles both;
            # each is independent, so one failing must not hide the other.
            out: dict[str, Any] = {}
            if args.family in (None, "trajectory"):
                from app.services.model_registry import ModelRegistry

                mv = ModelRegistry(session, settings).ensure_v1_bootstrap()
                out["trajectory"] = {"version": mv.version, "status": mv.status, "parent": mv.parent_version}
            if args.family in (None, "sea_ice"):
                out["sea_ice"] = _seaice_bootstrap(session, settings)
            _print(out)
        elif args.cmd == "ingest":
            from app.services.ingestion_service import IngestionService
            from app.services.usnic.source import FetchedDocument

            doc = None
            if args.file:
                content = args.file.read_bytes()
                doc = FetchedDocument(url=f"file://{args.file.resolve()}", discovery_method="manual_file", http_status=200,
                                      content_type="text/csv", content=content, fetched_at=datetime.now(UTC), duration_ms=0)
            o = IngestionService(session, settings).run(trigger="manual", document=doc)
            result: dict[str, Any] = {"icebergs": o.__dict__}
            # The sea-ice model is a core product with its own official feed.
            if args.skip_seaice:
                result["sea_ice"] = {"status": "skipped", "reason": "--skip-seaice"}
            elif not settings.seaice_enabled:
                result["sea_ice"] = {"status": "skipped", "reason": "SEAICE_ENABLED is false"}
            else:
                from app.services.seaice_ingestion_service import SeaIceIngestionService

                svc = SeaIceIngestionService(session, settings)
                ok, reason = svc.source.is_configured()
                result["sea_ice"] = svc.run(trigger="manual").as_dict() if ok else {"status": "skipped", "reason": reason}
            _print(result)
            return 0 if o.status != "failed" else 1
        elif args.cmd == "evaluate":
            from app.services.evaluation_service import EvaluationService

            _print(EvaluationService(session).evaluate().__dict__)
        elif args.cmd == "forecast":
            from app.services.forecast_service import ForecastService

            o = ForecastService(session, settings).run(trigger="manual")
            result: dict[str, Any] = {"trajectory": o.__dict__}
            # Two independent forecast products; each runs its OWN champion's
            # pipeline. The sea-ice U-Net is never substituted by the GRU.
            if args.skip_seaice:
                result["sea_ice"] = {"status": "skipped", "reason": "--skip-seaice"}
            elif not settings.seaice_enabled:
                result["sea_ice"] = {"status": "skipped", "reason": "SEAICE_ENABLED is false"}
            else:
                from app.services.seaice_forecast_service import SeaIceForecastService

                sea = SeaIceForecastService(session, settings).run(trigger="manual").as_dict()
                sea.pop("diagnostics", None)
                result["sea_ice"] = sea
            _print(result)
            return 0 if o.status not in ("failed",) else 1
        elif args.cmd == "retrain":
            from app.services.retraining_service import RetrainingService

            run = RetrainingService(session, settings).run(trigger="manual", force=args.force)
            _print({"status": run.status, "candidate": run.candidate_version, "champion": run.champion_version,
                    "decision": run.decision, "reason": run.failure_reason, "eligibility": run.eligibility})
        elif args.cmd == "benchmark":
            from app.services.retraining_service import benchmark_version

            _print(benchmark_version(session, settings, args.version))
        elif args.cmd == "status":
            from app.services.model_registry import ModelRegistry
            from app.services.retraining_service import RetrainingService

            reg = ModelRegistry(session, settings)
            champ = reg.get_deployed()
            _print({"deployed": champ.version if champ else None,
                    "deployed_feature_schema": champ.feature_schema_version if champ else None,
                    "versions": [(m.version, m.parent_version, m.status, m.feature_schema_version) for m in reg.list_versions()],
                    "retraining_eligibility": RetrainingService(session, settings).eligibility()})
        elif args.cmd == "env-status":
            from app.services.environment_service import EnvironmentService
            from app.services.retraining_service import RetrainingService

            env = EnvironmentService(session, settings)
            schemas, unavailable = RetrainingService(session, settings, environment=env).trainable_environmental_schemas()
            _print({"enabled": settings.env_enabled,
                    "sources": [{"group": s.group, "role": s.role, "provider": s.provider, "configured": s.configured,
                                 "reason": s.reason, "dataset_id": (s.spec or {}).get("dataset_id")} for s in env.status()],
                    "trainable_schemas": [x.version for x in schemas], "unavailable_schemas": unavailable,
                    "last_runs": env.last_runs()})
        elif args.cmd == "env-align":
            from app.services.environment_service import EnvironmentService

            _print(EnvironmentService(session, settings).align_observations())
        elif args.cmd == "env-overlay":
            from app.services.environment_service import EnvironmentService

            _print(EnvironmentService(session, settings).refresh_overlay())
        elif args.cmd == "seaice-register-base":
            from app.services.seaice_registry import SeaIceRegistry

            mv = SeaIceRegistry(session, settings).register_base()
            _print({"version": mv.version, "status": mv.status, "sha256": mv.artifact_sha256})
        elif args.cmd == "seaice-bootstrap":
            from app.services.seaice_registry import SeaIceRegistry

            mv = SeaIceRegistry(session, settings).ensure_champion()
            _print({"version": mv.version, "status": mv.status, "parent": mv.parent_version})
        elif args.cmd == "seaice-ingest":
            from app.services.seaice_ingestion_service import SeaIceIngestionService

            o = SeaIceIngestionService(session, settings).run(trigger="manual")
            _print(o.as_dict())
            return 0 if o.status != "failed" else 1
        elif args.cmd == "seaice-forecast":
            from app.services.seaice_forecast_service import SeaIceForecastService

            o = SeaIceForecastService(session, settings).run(trigger="manual")
            _print(o.as_dict())
            return 0 if o.status != "failed" else 1
        elif args.cmd == "seaice-evaluate":
            from app.services.seaice_evaluation_service import SeaIceEvaluationService

            _print(SeaIceEvaluationService(session, settings).run(trigger="manual").as_dict())
        elif args.cmd == "seaice-retrain":
            from app.services.seaice_retraining_service import SeaIceRetrainingService

            _print(SeaIceRetrainingService(session, settings).run(trigger="manual", force=args.force).as_dict())
        elif args.cmd == "seaice-status":
            from app.services.seaice_ingestion_service import SeaIceIngestionService
            from app.services.seaice_registry import SeaIceRegistry
            from app.services.seaice_retraining_service import SeaIceRetrainingService

            reg = SeaIceRegistry(session, settings)
            champion, latest = reg.get_champion(), reg.latest_version()
            _print({
                "champion": champion.version if champion else None,
                "latest_version": latest.version if latest else None,
                "champion_is_latest": bool(champion and latest and champion.version == latest.version),
                "versions": [(m.version, m.parent_version, m.status) for m in reg.list_versions()],
                "stored_observations": len(SeaIceIngestionService(session, settings).stored_dates()),
                "latest_observation": str(SeaIceIngestionService(session, settings).latest_stored_date()),
                "policy": SeaIceRetrainingService(session, settings).policy_snapshot(),
                "retraining_eligibility": SeaIceRetrainingService(session, settings).eligibility(),
            })
        elif args.cmd == "env-prefetch":
            from app.services.retraining_service import RetrainingService

            _print(RetrainingService(session, settings).prefetch_environment(args.max_samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
