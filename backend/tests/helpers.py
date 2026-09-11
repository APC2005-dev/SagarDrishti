"""Shared builders for integration tests (tiny synthetic model, observations, documents)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.models.tracking import Iceberg, Observation
from app.services.geo import ewkt_point
from app.services.usnic.source import FetchedDocument

HEADER = "Iceberg,Length (NM),Width (NM),Latitude,Longitude,Area (sqMI),Area (sqNM),Area (sqKM),Last Update"
BASE_METADATA = Path(__file__).resolve().parents[2] / "models" / "base" / "metadata.json"


def base_p90() -> dict[int, float]:
    """p90 radii recorded in the real models/base/metadata.json (tests copy that file)."""
    meta = json.loads(BASE_METADATA.read_text())
    return {int(h): v["p90_km"] for h, v in meta["metrics"]["by_horizon"].items() if v.get("p90_km") is not None}


def write_tiny_base(models_dir: Path) -> None:
    """A randomly-initialised model with the real architecture, saved in base layout.

    Used only so the pipeline can be exercised end to end; it is never the
    production artifact.
    """
    import joblib

    from ml.models.artifact_store import scaler_payload
    from ml.models.gru_architecture import build_model
    from ml.training.trainer import fit_scalers

    base = models_dir / "base"
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fs, ts = fit_scalers(rng.normal(size=(32, 14, 6)).astype(np.float32), rng.normal(size=(32, 14)).astype(np.float32))
    build_model().save(base / "global_gru_trajectory_model.keras")
    joblib.dump(scaler_payload(fs, ts), base / "global_gru_trajectory_scalers.joblib")
    metadata = json.loads((Path(__file__).resolve().parents[2] / "models" / "base" / "metadata.json").read_text())
    metadata["artifact_origin"] = "test_fixture"
    (base / "metadata.json").write_text(json.dumps(metadata))


def add_historical_track(session: Session, iceberg_id: str, start: date, days: int, lat0: float = -57.0, lon0: float = -47.0) -> None:
    if session.get(Iceberg, 1) is None or not session.query(Iceberg).filter_by(iceberg_id=iceberg_id).first():
        session.add(Iceberg(iceberg_id=iceberg_id, status="historical_only", first_seen=start, last_seen=start + timedelta(days=days - 1)))
        session.flush()
    for i in range(days):
        lat, lon = lat0 - 0.01 * i, lon0 + 0.02 * np.sin(i / 10)
        session.add(
            Observation(
                iceberg_id=iceberg_id, observation_date=start + timedelta(days=i), latitude=lat, longitude=lon,
                geom=ewkt_point(lat, lon), source="byu_consolidated_v8", provenance="historical_training_dataset",
                position_sensor="ascat", content_hash=f"h{i}", revision=1,
            )
        )
    session.commit()


def document(content: bytes | str, fetched_at: datetime | None = None) -> FetchedDocument:
    data = content.encode() if isinstance(content, str) else content
    return FetchedDocument(
        url="https://usicecenter.gov/File/DownloadCurrent?pId=134", discovery_method="configured", http_status=200,
        content_type="application/octet-stream", content=data, fetched_at=fetched_at or datetime(2026, 9, 11, tzinfo=UTC), duration_ms=12,
    )


def csv_rows(*rows: str) -> str:
    return HEADER + "\n" + "\n".join(rows) + "\n"
