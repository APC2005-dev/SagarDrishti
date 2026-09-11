"""Operator CLI (runs jobs synchronously, no broker required).

    python -m app.cli load-historical            # BYU dataset -> tracking.observations
    python -m app.cli register-base              # register models/base
    python -m app.cli bootstrap-v1               # create + deploy v1 from base
    python -m app.cli ingest [--file x.csv]      # USNIC fetch (or local file) -> DB
    python -m app.cli evaluate                   # score forecasts vs official observations
    python -m app.cli forecast                   # forecasts from the champion
    python -m app.cli retrain [--force]          # policy-gated retraining
    python -m app.cli benchmark --version v1     # all-horizon historical benchmark
    python -m app.cli status
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
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
    sub.add_parser("load-historical")
    sub.add_parser("register-base")
    sub.add_parser("bootstrap-v1")
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--file", type=Path, help="Import a local USNIC CSV instead of fetching")
    sub.add_parser("evaluate")
    sub.add_parser("forecast")
    p_rt = sub.add_parser("retrain")
    p_rt.add_argument("--force", action="store_true", help="Ignore eligibility thresholds (promotion policy still applies)")
    p_bm = sub.add_parser("benchmark")
    p_bm.add_argument("--version", required=True)
    sub.add_parser("status")
    sub.add_parser("env-status", help="environmental sources, credentials configured (yes/no), schemas trainable")
    p_ea = sub.add_parser("env-align", help="align environmental state to official observations")
    p_ea.add_argument("--all-latest", action="store_true", help="latest official fix of every active iceberg (default)")
    sub.add_parser("env-overlay", help="refresh coarse overlay grids for the map")
    p_ep = sub.add_parser("env-prefetch", help="pre-warm the environmental cache for a retraining experiment")
    p_ep.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        return _run(args, settings)
    except (ModelArtifactError, FileNotFoundError) as exc:
        # Expected operator errors (e.g. the base model files are not in models/base yet): one clear line, no traceback.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, settings: Any) -> int:
    with session_scope() as session:
        if args.cmd == "load-historical":
            from app.services.historical_loader import load_historical_dataset

            _print(load_historical_dataset(session, Path(settings.bootstrap_dataset_path)))
        elif args.cmd == "register-base":
            from app.services.model_registry import ModelRegistry

            mv = ModelRegistry(session, settings).register_base()
            _print({"version": mv.version, "status": mv.status, "sha256": mv.artifact_sha256})
        elif args.cmd == "bootstrap-v1":
            from app.services.model_registry import ModelRegistry

            mv = ModelRegistry(session, settings).ensure_v1_bootstrap()
            _print({"version": mv.version, "status": mv.status, "parent": mv.parent_version})
        elif args.cmd == "ingest":
            from app.services.ingestion_service import IngestionService
            from app.services.usnic.source import FetchedDocument

            doc = None
            if args.file:
                content = args.file.read_bytes()
                doc = FetchedDocument(url=f"file://{args.file.resolve()}", discovery_method="manual_file", http_status=200,
                                      content_type="text/csv", content=content, fetched_at=datetime.now(UTC), duration_ms=0)
            o = IngestionService(session, settings).run(trigger="manual", document=doc)
            _print(o.__dict__)
            return 0 if o.status != "failed" else 1
        elif args.cmd == "evaluate":
            from app.services.evaluation_service import EvaluationService

            _print(EvaluationService(session).evaluate().__dict__)
        elif args.cmd == "forecast":
            from app.services.forecast_service import ForecastService

            o = ForecastService(session, settings).run(trigger="manual")
            _print(o.__dict__)
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
        elif args.cmd == "env-prefetch":
            from app.services.retraining_service import RetrainingService

            _print(RetrainingService(session, settings).prefetch_environment(args.max_samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
