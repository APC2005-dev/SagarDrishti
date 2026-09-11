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
        )
    ]
