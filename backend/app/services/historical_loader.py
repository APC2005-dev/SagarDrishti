"""Load the historical training dataset (BYU v8.0) into ``tracking.observations``.

Rows get provenance ``historical_training_dataset`` — they are available to the
sequence adapter as bootstrap history but are never used as ground truth for
operational evaluation. Loading is idempotent (ON CONFLICT DO NOTHING).
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.tracking import Iceberg, Observation
from app.services.geo import ewkt_point
from ml.adapters.bootstrap_adapter import DATASET_NAME, load_byu_tracks, sha256_file

log = get_logger(__name__)
SOURCE_NAME = "byu_consolidated_v8"
PROVENANCE = "historical_training_dataset"


def _clean(v: object) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)  # type: ignore[arg-type]


def load_historical_dataset(session: Session, zip_path: Path, chunk_size: int = 5000) -> dict[str, int | str]:
    if not zip_path.exists():
        raise FileNotFoundError(f"historical dataset not found: {zip_path}")
    digest = sha256_file(zip_path)
    tracks = load_byu_tracks(zip_path)
    log.info("historical_load_started", rows=len(tracks), icebergs=int(tracks["iceberg_id"].nunique()), sha256=digest)

    spans = tracks.groupby("iceberg_id")["date"].agg(["min", "max"])
    existing = set(session.execute(select(Iceberg.iceberg_id)).scalars())
    now = datetime.now(UTC)
    for iceberg_id, span in spans.iterrows():
        if iceberg_id in existing:
            continue
        session.add(
            Iceberg(
                iceberg_id=iceberg_id,
                status="historical_only",
                first_seen=span["min"].date(),
                last_seen=span["max"].date(),
                status_changed_at=now,
            )
        )
    session.flush()
    # widen first/last seen for icebergs that also have official observations
    for iceberg in session.execute(select(Iceberg).where(Iceberg.iceberg_id.in_(spans.index.tolist()))).scalars():
        lo, hi = spans.loc[iceberg.iceberg_id, "min"].date(), spans.loc[iceberg.iceberg_id, "max"].date()
        iceberg.first_seen = min(filter(None, [iceberg.first_seen, lo]))
        iceberg.last_seen = max(filter(None, [iceberg.last_seen, hi]))

    before = session.execute(select(func.count()).select_from(Observation).where(Observation.provenance == PROVENANCE)).scalar_one()
    rows = []
    for rec in tracks.itertuples(index=False):
        lat, lon = float(rec.latitude), float(rec.longitude)
        h = hashlib.sha256(f"{lat:.6f},{lon:.6f},{rec.position_source}".encode()).hexdigest()
        rows.append(
            {
                "iceberg_id": rec.iceberg_id,
                "observation_date": rec.date.date(),
                "latitude": lat,
                "longitude": lon,
                "geom": ewkt_point(lat, lon),
                "source": SOURCE_NAME,
                "source_file": rec.source_file,
                "provenance": PROVENANCE,
                "position_sensor": rec.position_source,
                "raw_attributes": {"size_1": _clean(rec.size_1), "size_2": _clean(rec.size_2), "dataset": DATASET_NAME, "dataset_sha256": digest},
                "content_hash": h,
                "revision": 1,
            }
        )
        if len(rows) >= chunk_size:
            session.execute(insert(Observation).on_conflict_do_nothing(constraint="uq_observations_iceberg_date_provenance"), rows)
            rows = []
    if rows:
        session.execute(insert(Observation).on_conflict_do_nothing(constraint="uq_observations_iceberg_date_provenance"), rows)
    after = session.execute(select(func.count()).select_from(Observation).where(Observation.provenance == PROVENANCE)).scalar_one()
    session.commit()
    result = {"dataset": DATASET_NAME, "sha256": digest, "rows_in_dataset": len(tracks), "inserted": after - before, "total_historical": after}
    log.info("historical_load_completed", **result)
    return result
