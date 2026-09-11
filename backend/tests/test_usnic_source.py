from __future__ import annotations

import httpx
import pytest

from app.services.usnic.source import (
    SourceUnavailableError,
    SourceValidationError,
    UsnicSourceClient,
    discover_csv_url,
    validate_payload,
)

CSV = b"Iceberg,Length (NM),Width (NM),Latitude,Longitude,Area (sqMI),Area (sqNM),Area (sqKM),Last Update\nA81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026\n"
PAGE = """
<html><body>
<a href="/File/DownloadCurrent?pId=135">PDF</a>
<a class="btn" href="/File/DownloadCurrent?pId=999"><span>CSV</span></a>
<a href="/File/DownloadCurrent?pId=228">Shapefile</a>
</body></html>
"""


def test_discover_csv_link() -> None:
    assert discover_csv_url(PAGE, "https://usicecenter.gov/Products/AntarcIcebergs") == "https://usicecenter.gov/File/DownloadCurrent?pId=999"
    assert discover_csv_url("<html>no links</html>", "https://x") is None


@pytest.mark.parametrize(
    "content,ctype,match",
    [(b"", "text/csv", "empty"), (b"<!DOCTYPE html><html>", "text/html", "content-type|HTML"),
     (b"<html><head>", "application/octet-stream", "HTML"), (b"x" * 100, "text/csv", "exceeds"),
     (b"no commas here\n", "text/csv", "header"), (CSV, "image/png", "content-type")],
)
def test_validate_payload_rejects(content: bytes, ctype: str, match: str) -> None:
    with pytest.raises(SourceValidationError, match=match):
        validate_payload(content, ctype, max_bytes=50 if match == "exceeds" else 10_000)


def test_validate_payload_accepts_octet_stream() -> None:
    validate_payload(CSV, "application/octet-stream", 10_000)


def _client(settings, handler) -> UsnicSourceClient:  # type: ignore[no-untyped-def]
    s = settings.model_copy(update={"usnic_retries": 0})
    return UsnicSourceClient(s, httpx.Client(transport=httpx.MockTransport(handler)))


def test_fetch_configured_url(settings) -> None:  # type: ignore[no-untyped-def]
    client = _client(settings, lambda r: httpx.Response(200, content=CSV, headers={"content-type": "application/octet-stream"}))
    doc = client.fetch()
    assert doc.discovery_method == "configured" and doc.content == CSV and doc.http_status == 200


def test_fetch_falls_back_to_discovery(settings) -> None:  # type: ignore[no-untyped-def]
    def handler(req: httpx.Request) -> httpx.Response:
        if "pId=134" in str(req.url):
            return httpx.Response(404)
        if "AntarcIcebergs" in str(req.url):
            return httpx.Response(200, text=PAGE)
        if "pId=999" in str(req.url):
            return httpx.Response(200, content=CSV, headers={"content-type": "text/csv"})
        return httpx.Response(500)

    doc = _client(settings, handler).fetch()
    assert doc.discovery_method == "discovered" and doc.url.endswith("pId=999")


def test_fetch_unavailable(settings) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SourceUnavailableError):
        _client(settings, lambda r: httpx.Response(503)).fetch()


def test_network_error(settings) -> None:  # type: ignore[no-untyped-def]
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(SourceUnavailableError):
        _client(settings, boom).fetch()
