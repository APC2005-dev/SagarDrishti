from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from app.services.status_service import feed_state, is_stale

NOW = datetime(2026, 9, 11, 12, tzinfo=UTC)


def state(status, new=0, success_hours_ago=1.0, latest=date(2026, 9, 10)):  # type: ignore[no-untyped-def]
    success = NOW - timedelta(hours=success_hours_ago) if success_hours_ago is not None else None
    return feed_state(status, new, success, latest, degraded_after_hours=96, stale_after_days=10, now=NOW)[0]


def test_states() -> None:
    assert state(None) == "UNKNOWN"
    assert state("success", new=33) == "LIVE"
    assert state("unchanged") == "SYNCED"
    assert state("success", new=0) == "SYNCED"
    assert state("failed") == "DEGRADED"
    assert state("partial", new=5) == "DEGRADED"
    assert state("failed", success_hours_ago=200) == "FAILED"
    assert state("failed", success_hours_ago=None) == "FAILED"
    assert state("success", new=3, latest=date(2026, 8, 1)) == "DEGRADED"


def test_is_stale() -> None:
    assert not is_stale(date(2026, 9, 5), 10, today=date(2026, 9, 11))
    assert is_stale(date(2026, 8, 20), 10, today=date(2026, 9, 11))
    assert is_stale(None, 10)
