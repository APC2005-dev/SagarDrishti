from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Path, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_async_session
from ml.adapters.bootstrap_adapter import normalize_iceberg_id

SessionDep = Annotated[AsyncSession, Depends(get_async_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
Limit = Annotated[int, Query(ge=1, le=500, description="Page size")]
Offset = Annotated[int, Query(ge=0, description="Page offset")]
Horizon = Annotated[int, Query(ge=1, le=7, description="Return horizons 1..h of the same 7-day model run (1, 3 or 7 in the UI)")]


def iceberg_path(iceberg_id: Annotated[str, Path(min_length=2, max_length=32, description="USNIC designator, e.g. A23A")]) -> str:
    normalized = normalize_iceberg_id(iceberg_id)
    if not normalized:
        raise HTTPException(status_code=422, detail="invalid iceberg id")
    return normalized


IcebergId = Annotated[str, Depends(iceberg_path)]
