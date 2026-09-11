"""Engines and sessions.

* The API uses an async engine (asyncpg) for read endpoints.
* Workers, CLI and Alembic use a sync engine (psycopg) — ML jobs are CPU-bound
  and run outside the request path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache(maxsize=1)
def sync_engine() -> Engine:
    return create_engine(get_settings().sync_database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


@lru_cache(maxsize=1)
def async_engine() -> AsyncEngine:
    return create_async_engine(get_settings().async_database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)


@lru_cache(maxsize=1)
def _sync_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=sync_engine(), expire_on_commit=False)


@lru_cache(maxsize=1)
def _async_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=async_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for workers: commit on success, rollback on error."""
    session = _sync_factory()()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with _async_factory()() as session:
        yield session
