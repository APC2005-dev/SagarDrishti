"""Test configuration.

Environment is prepared *before* the app is imported: DATABASE_URL points at
TEST_DATABASE_URL (a separate database, created and migrated automatically),
and model/raw-data directories are redirected to a temp folder so tests never
touch the real ``models/`` store.

Tests marked ``db`` are skipped when no PostGIS database is reachable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
load_dotenv(REPO / ".env")
TEST_DB_URL = os.environ.get("TEST_DATABASE_URL")
os.environ["DATABASE_URL"] = TEST_DB_URL or "postgresql+psycopg://test:test@localhost:1/unavailable"
_TMP = Path(tempfile.mkdtemp(prefix="sagar-tests-"))
os.environ["MODELS_DIR"] = str(_TMP / "models")
os.environ["RAW_DATA_DIR"] = str(_TMP / "raw")
os.environ["BASEMAP_CACHE_DIR"] = str(_TMP / "basemap")
# Environmental tests enable the pipeline explicitly with synthetic providers; nothing
# in the suite may reach Copernicus / CDS even if real credentials are present in .env.
os.environ["ENV_ENABLED"] = "false"
os.environ["ENV_CACHE_DIR"] = str(_TMP / "environment")
for _cred in ("COPERNICUS_MARINE_USERNAME", "COPERNICUS_MARINE_PASSWORD", "COPERNICUSMARINE_SERVICE_USERNAME",
              "COPERNICUSMARINE_SERVICE_PASSWORD", "CDSAPI_KEY", "CDS_API_KEY"):
    os.environ[_cred] = ""
os.environ["LOG_JSON"] = "false"
(_TMP / "models").mkdir()

import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import Engine, make_url  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import get_settings  # noqa: E402

_db_state: dict[str, object] = {}


def _ensure_database(url: str) -> Engine | None:
    try:
        u = make_url(url)
        admin = create_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            if not c.execute(text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": u.database}).scalar():
                c.execute(text(f'CREATE DATABASE "{u.database}"'))
        admin.dispose()
        engine = create_engine(url)
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # noqa: BLE001
        _db_state["error"] = f"{type(exc).__name__}: {exc}"
        return None


def _migrate() -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(REPO / "backend" / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO / "backend" / "alembic"))
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def engine() -> Engine:
    if "engine" not in _db_state:
        eng = _ensure_database(TEST_DB_URL) if TEST_DB_URL else None
        if eng is not None:
            _migrate()
        _db_state["engine"] = eng
    eng = _db_state["engine"]
    if eng is None:
        pytest.skip(f"PostGIS test database unavailable ({_db_state.get('error', 'TEST_DATABASE_URL not set')})")
    return eng  # type: ignore[return-value]


def _truncate(engine: Engine) -> None:
    with engine.begin() as c:
        tables = c.execute(
            text("SELECT schemaname || '.' || tablename FROM pg_tables WHERE schemaname IN ('tracking', 'ml')")
        ).scalars().all()
        if tables:
            c.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))


@pytest.fixture
def db(engine: Engine) -> Session:
    _truncate(engine)
    import shutil
    import stat

    models = Path(get_settings().models_dir)
    for child in models.iterdir():
        for f in child.rglob("*"):
            os.chmod(f, stat.S_IWUSR | stat.S_IRUSR)
        shutil.rmtree(child)
    with Session(engine, expire_on_commit=False) as session:
        yield session


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def fixture_csv() -> bytes:
    return (FIXTURES / "usnic_antarctic_icebergs_2026-09-10.csv").read_bytes()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "backend" in str(item.fspath) and ("test_db" in item.name or "db" in item.fixturenames or "engine" in item.fixturenames):
            item.add_marker(pytest.mark.db)
