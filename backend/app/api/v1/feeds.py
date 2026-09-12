"""Data source registry. Only USNIC is live today; the list is data-driven so
future sources (sea-ice concentration, ocean currents, AIS ...) add an entry."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import SessionDep, SettingsDep
from app.repositories import queries
from app.schemas.operations import FeedOut
from app.services.status_service import feed_state

router = APIRouter(prefix="/feeds", tags=["feeds"])


@router.get("", response_model=list[FeedOut], summary="Status of every configured data source")
async def feeds(session: SessionDep, settings: SettingsDep) -> list[FeedOut]:
    runs = await queries.ingestion_runs(session, 1)
    latest = runs[0] if runs else None
    success = await queries.last_successful_ingestion(session)
    latest_date = await queries.latest_official_date(session)
    state, reasons = feed_state(
        latest.status if latest else None, latest.new_observations if latest else 0,
        success.fetched_at if success else None, latest_date, settings.feed_degraded_after_hours, settings.stale_after_days,
    )
    return [
        FeedOut(
            id="usnic_antarctic_icebergs",
            name="USNIC Antarctic Iceberg Feed",
            provider="U.S. National Ice Center",
            description="Official positions and dimensions of named Antarctic icebergs (CSV). Analysed weekly; polled every "
            f"{settings.ingestion_interval_hours:.0f} h. Observation date = 'Last Update'.",
            product_url=settings.usnic_product_page_url,
            source_url=latest.source_url if latest else settings.usnic_csv_url,
            cadence="weekly (source) / polled every 72 h",
            poll_interval_hours=settings.ingestion_interval_hours,
            state=state,
            state_reasons=reasons,
            last_fetch_at=latest.fetched_at if latest else None,
            last_success_at=success.fetched_at if success else None,
            latest_official_observation_date=latest_date,
            fetch_duration_ms=latest.duration_ms if latest else None,
            checksum_sha256=latest.checksum_sha256 if latest else None,
            record_count=latest.row_count if latest else None,
            discovery_method=latest.discovery_method if latest else None,
            error_message=latest.error_message if latest else None,
        ),
        *_seaice_feed(settings),
        *_environmental_feeds(settings),
    ]


def _seaice_feed(settings) -> list[FeedOut]:  # type: ignore[no-untyped-def]
    """The sea-ice CORE feed: the exact Copernicus Marine product the sea-ice
    model was trained on. Distinct from the environmental ``sea_ice`` feature
    provider below, which samples a different (model) product per iceberg."""
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from app.db.session import sync_engine
    from app.models.seaice import SeaIceObservation, SeaIceRun
    from ml.seaice.constants import AUTHORITY, DATASET_ID, VARIABLE

    if not settings.seaice_enabled:
        return []
    with Session(sync_engine()) as session:
        latest = session.execute(
            select(SeaIceRun).where(SeaIceRun.kind == "ingestion")
            .order_by(SeaIceRun.started_at.desc()).limit(1)
        ).scalar_one_or_none()
        ok_at = session.execute(
            select(func.max(SeaIceRun.completed_at)).where(
                SeaIceRun.kind == "ingestion", SeaIceRun.status.in_(("success", "unchanged"))
            )
        ).scalar()
        count, newest = session.execute(
            select(func.count(), func.max(SeaIceObservation.observation_date)).select_from(SeaIceObservation)
        ).one()
        from app.services.seaice_ingestion_service import SeaIceIngestionService

        configured, reason = SeaIceIngestionService(session, settings).source.is_configured()

    state, reasons = feed_state(
        latest.status if latest else None,
        latest.entries_new if latest else 0,
        ok_at, newest, settings.feed_degraded_after_hours, settings.stale_after_days,
    )
    if not configured:
        state, reasons = "UNKNOWN", [reason]
    return [
        FeedOut(
            id="copernicus_seaice_osisaf",
            name="Copernicus Marine / OSI SAF Antarctic Sea-Ice Concentration",
            provider=AUTHORITY,
            description=(
                f"Official daily sea-ice concentration ({VARIABLE}, %) on a 0.1 deg grid, coarsened 5x to "
                f"0.5 deg (100x720). The exact product the sea-ice U-Net was trained on. Polled every "
                f"{settings.seaice_poll_interval_hours:.0f} h; a day already stored is never re-ingested."
            ),
            product_url="https://data.marine.copernicus.eu/product/SEAICE_GLO_SEAICE_L4_NRT_OBSERVATIONS_011_001",
            source_url=DATASET_ID,
            cadence="P1D native, daily values; latency ~1-2 d",
            poll_interval_hours=settings.seaice_poll_interval_hours,
            state=state,
            state_reasons=reasons,
            last_fetch_at=latest.started_at if latest else None,
            last_success_at=ok_at,
            latest_official_observation_date=newest,
            fetch_duration_ms=latest.duration_ms if latest else None,
            checksum_sha256=None,  # per-day grids are checksummed individually, not the feed as a whole
            record_count=int(count),
            discovery_method="configured dataset_id",
            error_message=latest.error_message if latest else None,
            category="sea_ice",
            configured=configured,
        )
    ]


def _environmental_feeds(settings) -> list[FeedOut]:  # type: ignore[no-untyped-def]
    """One entry per configured source (operational and historical). Unconfigured
    sources are listed with state UNKNOWN and the reason — never with invented data."""
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from app.db.session import sync_engine
    from app.models.environment import EnvCacheEntry, EnvIngestionRun
    from app.services.environment_service import EnvironmentService

    feeds: list[FeedOut] = []
    with Session(sync_engine()) as session:
        env = EnvironmentService(session, settings)
        for st in env.status():
            if st.provider is None or st.spec is None:
                continue
            spec = st.spec
            last = session.execute(
                select(EnvIngestionRun).where(EnvIngestionRun.provider == st.provider).order_by(EnvIngestionRun.started_at.desc()).limit(1)
            ).scalar_one_or_none()
            ok_at = session.execute(
                select(func.max(EnvIngestionRun.completed_at)).where(EnvIngestionRun.provider == st.provider,
                                                                     EnvIngestionRun.status.in_(("success", "partial")))
            ).scalar()
            n, newest = session.execute(
                select(func.count(), func.max(EnvCacheEntry.max_valid_date)).where(EnvCacheEntry.provider == st.provider)
            ).one()
            if not st.configured:
                state, reasons = "UNKNOWN", [f"not configured: {st.reason}"]
            elif last is None:
                state, reasons = "UNKNOWN", ["no fetch yet"]
            elif last.status == "failed":
                recent = ok_at is not None and datetime.now(UTC) - ok_at < timedelta(hours=settings.feed_degraded_after_hours)
                state, reasons = ("DEGRADED" if recent else "FAILED"), [last.error_message or "last fetch failed"]
            else:
                state, reasons = "SYNCED", []
            feeds.append(FeedOut(
                id=f"env_{st.role}_{st.group}", name=f"{spec['authority']} — {st.group.replace('_', ' ')} ({st.role})",
                provider=spec["authority"],
                description=f"{spec['product_id']} / {spec['dataset_id']}. {spec['aggregation']}. "
                            f"{('Depth: ' + spec['depth'] + '. ') if spec.get('depth') else ''}{spec.get('notes') or ''}",
                product_url=(f"https://data.marine.copernicus.eu/product/{spec['product_id']}/description"
                             if spec["authority"].startswith("Copernicus Marine")
                             else "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"),
                source_url=spec["dataset_id"],
                cadence=f"{spec['temporal_resolution']} native, daily values; latency ~{spec['latency_days']:g} d",
                poll_interval_hours=24.0,
                state=state,  # type: ignore[arg-type]
                state_reasons=reasons,
                last_fetch_at=last.started_at if last else None,
                last_success_at=ok_at,
                latest_official_observation_date=newest,
                fetch_duration_ms=last.duration_ms if last else None,
                checksum_sha256=None,
                record_count=int(n),
                discovery_method=st.role,
                error_message=last.error_message if last and last.status == "failed" else None,
                category="environmental",
                configured=st.configured,
            ))
    return feeds
