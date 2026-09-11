"""Fetching the official USNIC Antarctic iceberg CSV.

The configured URL is tried first. If it fails or returns something that is not
a CSV (the product has moved before), the product page is scraped for its
current CSV link and that is tried instead. The discovery method used is stored
on the ingestion run, so an operator can see when the source moved.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin

import httpx

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.usnic.parser import decode_csv_bytes, looks_like_html

log = get_logger(__name__)

_CSV_LINK = re.compile(
    r"<a[^>]+href=[\"']([^\"']*DownloadCurrent\?pId=\d+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL
)
ACCEPTED_CONTENT_TYPES = ("text/csv", "application/octet-stream", "text/plain", "application/csv", "application/vnd.ms-excel")


class SourceUnavailableError(RuntimeError):
    """Network failure or non-200 response from every candidate URL."""


class SourceValidationError(RuntimeError):
    """A response arrived but is not an acceptable CSV payload."""


@dataclass(frozen=True)
class FetchedDocument:
    url: str
    discovery_method: str
    http_status: int
    content_type: str | None
    content: bytes
    fetched_at: datetime
    duration_ms: int


def discover_csv_url(page_html: str, base_url: str) -> str | None:
    """Return the CSV link from the product page (link text mentions CSV)."""
    for href, label in _CSV_LINK.findall(page_html):
        if "csv" in re.sub(r"<[^>]+>", " ", label).lower():
            return urljoin(base_url, href)
    return None


def validate_payload(content: bytes, content_type: str | None, max_bytes: int) -> None:
    if not content:
        raise SourceValidationError("empty response body")
    if len(content) > max_bytes:
        raise SourceValidationError(f"response of {len(content)} bytes exceeds limit {max_bytes}")
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype and ctype not in ACCEPTED_CONTENT_TYPES and not ctype.startswith("text/"):
        raise SourceValidationError(f"unexpected content-type {ctype!r}")
    text = decode_csv_bytes(content)
    if looks_like_html(text):
        raise SourceValidationError("response is HTML, not CSV")
    if "," not in text.splitlines()[0]:
        raise SourceValidationError("first line is not a comma-separated header")


class UsnicSourceClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self._client = client or httpx.Client(
            timeout=settings.usnic_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": settings.usnic_user_agent},
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(self.settings.usnic_retries + 1):
            try:
                resp = self._client.get(url)
                if resp.status_code >= 500 and attempt < self.settings.usnic_retries:
                    time.sleep(2**attempt)
                    continue
                return resp
            except httpx.HTTPError as exc:
                last = exc
                if attempt < self.settings.usnic_retries:
                    time.sleep(2**attempt)
        raise SourceUnavailableError(f"GET {url} failed: {last}")

    def _fetch_csv(self, url: str, method: str) -> FetchedDocument:
        started = time.monotonic()
        fetched_at = datetime.now(UTC)
        resp = self._get(url)
        if resp.status_code != 200:
            raise SourceUnavailableError(f"GET {url} returned HTTP {resp.status_code}")
        content_type = resp.headers.get("content-type")
        validate_payload(resp.content, content_type, self.settings.usnic_max_bytes)
        return FetchedDocument(
            url=str(resp.url),
            discovery_method=method,
            http_status=resp.status_code,
            content_type=content_type,
            content=resp.content,
            fetched_at=fetched_at,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def fetch(self) -> FetchedDocument:
        try:
            return self._fetch_csv(self.settings.usnic_csv_url, "configured")
        except (SourceUnavailableError, SourceValidationError) as primary:
            log.warning("usnic_configured_url_failed", url=self.settings.usnic_csv_url, error=str(primary))
            page = self._get(self.settings.usnic_product_page_url)
            if page.status_code != 200:
                raise SourceUnavailableError(
                    f"configured URL failed ({primary}); product page HTTP {page.status_code}"
                ) from primary
            discovered = discover_csv_url(page.text, str(page.url))
            if not discovered:
                raise SourceUnavailableError(f"configured URL failed ({primary}); no CSV link on product page") from primary
            if discovered == self.settings.usnic_csv_url:
                raise primary
            log.warning("usnic_source_discovered", url=discovered)
            return self._fetch_csv(discovered, "discovered")
