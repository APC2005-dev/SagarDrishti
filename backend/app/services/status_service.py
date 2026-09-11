"""Derivation of operator-facing states (LIVE / SYNCED / DEGRADED / FAILED / UNKNOWN)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Literal

FeedState = Literal["LIVE", "SYNCED", "DEGRADED", "FAILED", "UNKNOWN"]


def feed_state(
    latest_status: str | None,
    latest_new_observations: int,
    last_success_at: datetime | None,
    latest_official_date: date | None,
    degraded_after_hours: float,
    stale_after_days: int,
    now: datetime | None = None,
) -> tuple[FeedState, list[str]]:
    """
    LIVE      latest fetch succeeded and brought new official observations
    SYNCED    latest fetch succeeded, nothing new (normal between weekly USNIC updates)
    DEGRADED  a recent success exists but the latest fetch failed/partially failed,
              or the last success is too old, or the official data itself is stale
    FAILED    no successful fetch within the degraded window
    UNKNOWN   no ingestion has ever run
    """
    now = now or datetime.now(UTC)
    reasons: list[str] = []
    if latest_status is None:
        return "UNKNOWN", ["no ingestion has run yet"]
    recent_success = last_success_at is not None and now - last_success_at <= timedelta(hours=degraded_after_hours)
    if not recent_success:
        return "FAILED", [f"no successful fetch in the last {degraded_after_hours:.0f} h"]
    state: FeedState = "LIVE" if latest_status == "success" and latest_new_observations > 0 else "SYNCED"
    if latest_status == "failed":
        state, reasons = "DEGRADED", ["latest fetch failed; serving last successful data"]
    elif latest_status == "partial":
        state, reasons = "DEGRADED", ["latest file had rejected rows"]
    if latest_official_date is not None and (now.date() - latest_official_date).days > stale_after_days:
        state = "DEGRADED"
        reasons.append(f"latest official observation {latest_official_date} is older than {stale_after_days} days")
    return state, reasons


def is_stale(observation_date: date | None, stale_after_days: int, today: date | None = None) -> bool:
    if observation_date is None:
        return True
    return ((today or datetime.now(UTC).date()) - observation_date).days > stale_after_days
