"""Port search, resolution and validation against the NGA World Port Index.

The World Port Index is the authoritative source; this service is the only thing
that talks to it. The browser never does — it calls Sagar Drishti, which resolves
against the cached index in ``routing.ports``.

A station, base or arbitrary geographic location is **not** a port: only entries
present in the WPI resolve. There is no generic geocoding fallback.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.routing import Port
from app.services.geo import ewkt_point

log = get_logger(__name__)

SOURCE = "nga_world_port_index"
_DMS = re.compile(r"^\s*(\d+)\D+(\d+)'(?:\s*(\d+(?:\.\d+)?)\D*)?\s*([NSEW])")


class PortError(Exception):
    """Machine-readable port failure; ``code`` is what the API returns."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ResolvedPort:
    id: int
    identifier: str
    name: str
    latitude: float
    longitude: float
    country_name: str | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "identifier": self.identifier,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country_name": self.country_name,
            "source": self.source,
        }


def normalize(name: str) -> str:
    """Fold case, accents and punctuation so ``St. John's`` matches ``st johns``."""
    folded = unicodedata.normalize("NFKD", str(name))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    # Apostrophes are dropped, not turned into separators, so "St. John's"
    # matches "St Johns"; every other punctuation run becomes a single space.
    folded = folded.lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def parse_coordinate(value: Any) -> float | None:
    """WPI serves either decimal degrees or ``62°05'00"S``-style DMS."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = _DMS.match(text)
    if not match:
        return None
    degrees, minutes, seconds, hemisphere = match.group(1), match.group(2), match.group(3) or 0, match.group(4)
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    return -decimal if hemisphere in ("S", "W") else decimal


class PortResolver:
    def __init__(self, session: Session, settings: Settings, client: httpx.Client | None = None) -> None:
        self.session = session
        self.settings = settings
        self._client = client

    # ------------------------------------------------------------------ cache
    def cached_count(self) -> int:
        return int(self.session.execute(select(func.count()).select_from(Port)).scalar_one())

    def index_is_stale(self) -> bool:
        newest = self.session.execute(select(func.max(Port.refreshed_at))).scalar()
        if newest is None:
            return True
        return datetime.now(UTC) - newest > timedelta(days=self.settings.wpi_refresh_after_days)

    def refresh_index(self, force: bool = False) -> dict[str, int]:
        """Fetch the World Port Index and upsert it into ``routing.ports``.

        Called on demand, not per search: the index changes on a publication
        cycle, so repeated searches never hit NGA.
        """
        if not force and not self.index_is_stale():
            return {"fetched": 0, "inserted": 0, "updated": 0, "cached": self.cached_count()}
        owned = self._client is None
        client = self._client or httpx.Client(timeout=self.settings.wpi_timeout_seconds, follow_redirects=True)
        try:
            response = client.get(self.settings.wpi_api_url, params={"output": "json"})
            response.raise_for_status()
            rows = response.json().get("ports", [])
        except Exception as exc:  # noqa: BLE001 - surfaced as a recoverable API error
            raise PortError(
                "PORT_SERVICE_UNAVAILABLE",
                f"The World Port Index service could not be reached ({type(exc).__name__}).",
            ) from exc
        finally:
            if owned:
                client.close()

        existing = {p.identifier: p for p in self.session.execute(select(Port)).scalars()}
        inserted = updated = skipped = 0
        now = datetime.now(UTC)
        for row in rows:
            identifier = str(row.get("portNumber") or "").strip()
            name = str(row.get("portName") or "").strip()
            latitude = parse_coordinate(row.get("latitude"))
            longitude = parse_coordinate(row.get("longitude"))
            if not identifier or not name or latitude is None or longitude is None:
                skipped += 1  # unusable row: never invent a coordinate for it
                continue
            port = existing.get(identifier)
            if port is None:
                port = Port(source=SOURCE, identifier=identifier)
                self.session.add(port)
                inserted += 1
            else:
                updated += 1
            port.name = name
            port.normalized_name = normalize(name)
            port.unlocode = (row.get("unloCode") or None) or None
            port.country_code = (row.get("countryCode") or None) or None
            port.country_name = (row.get("countryName") or None) or None
            port.region_name = (row.get("regionName") or None) or None
            port.latitude = latitude
            port.longitude = longitude
            port.geom = ewkt_point(latitude, longitude)
            port.attributes = {
                k: row.get(k)
                for k in ("harborSize", "harborType", "shelter", "navArea", "chartNumber")
                if row.get(k) is not None
            }
            port.refreshed_at = now
        self.session.flush()
        log.info("wpi_index_refreshed", fetched=len(rows), inserted=inserted, updated=updated, skipped=skipped)
        return {"fetched": len(rows), "inserted": inserted, "updated": updated, "skipped": skipped,
                "cached": self.cached_count()}

    def ensure_index(self) -> None:
        if self.cached_count() == 0:
            self.refresh_index(force=True)

    # ----------------------------------------------------------------- search
    def search(self, query: str, limit: int = 10, domain_only: bool = False) -> list[ResolvedPort]:
        """Ports whose name matches ``query``; exact, then prefix, then substring."""
        text = normalize(query)
        if not text:
            return []
        self.ensure_index()
        statement = select(Port).where(Port.normalized_name.like(f"%{text}%"))
        if domain_only:
            statement = statement.where(
                Port.longitude >= self.settings.routing_domain_west,
                Port.longitude <= self.settings.routing_domain_east,
                Port.latitude >= self.settings.routing_domain_south,
                Port.latitude <= self.settings.routing_domain_north,
            )
        rows = list(self.session.execute(statement.limit(200)).scalars())

        def rank(port: Port) -> tuple[int, int, str]:
            name = port.normalized_name
            kind = 0 if name == text else 1 if name.startswith(text) else 2
            return (kind, len(name), name)

        return [self._resolved(p) for p in sorted(rows, key=rank)[:limit]]

    # --------------------------------------------------------------- resolving
    def resolve(self, value: str | None, role: str) -> ResolvedPort:
        """Resolve a canonical identifier or a name. ``role`` is departure/destination."""
        invalid = "INVALID_DEPARTURE_PORT" if role == "departure" else "INVALID_DESTINATION_PORT"
        if value is None or not str(value).strip():
            raise PortError(invalid, f"{role.capitalize()} port is required.")
        value = str(value).strip()
        self.ensure_index()
        port = self.session.execute(
            select(Port).where(Port.source == SOURCE, Port.identifier == value)
        ).scalar_one_or_none()
        if port is None:
            text = normalize(value)
            candidates = list(
                self.session.execute(select(Port).where(Port.normalized_name == text)).scalars()
            )
            if len(candidates) == 1:
                port = candidates[0]
            elif len(candidates) > 1:
                # Ambiguous alias: make the caller choose a canonical identifier
                # rather than guessing which port was meant.
                names = ", ".join(f"{c.name} ({c.country_name}) [{c.identifier}]" for c in candidates[:5])
                raise PortError(
                    "PORT_NOT_FOUND",
                    f"'{value}' matches more than one port in the World Port Index: {names}. "
                    "Select one from the suggestions.",
                )
        if port is None:
            raise PortError(
                "PORT_NOT_FOUND", "The specified port could not be found in the World Port Index."
            )
        return self._resolved(port)

    def validate_routing_domain(self, port: ResolvedPort) -> None:
        s = self.settings
        inside = (
            s.routing_domain_west <= port.longitude <= s.routing_domain_east
            and s.routing_domain_south <= port.latitude <= s.routing_domain_north
        )
        if not inside:
            raise PortError(
                "PORT_OUTSIDE_ROUTING_DOMAIN",
                f"{port.name} is outside the currently supported Antarctic routing region "
                f"(longitude {s.routing_domain_west}..{s.routing_domain_east}, "
                f"latitude {s.routing_domain_south}..{s.routing_domain_north}).",
            )

    @staticmethod
    def _resolved(port: Port) -> ResolvedPort:
        return ResolvedPort(
            id=port.id,
            identifier=port.identifier,
            name=port.name,
            latitude=float(port.latitude),
            longitude=float(port.longitude),
            country_name=port.country_name,
            source=port.source,
        )
