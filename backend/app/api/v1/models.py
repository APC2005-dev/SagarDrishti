from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import Limit, SessionDep
from app.api.serializers import evaluation_out
from app.repositories import queries
from app.schemas.common import ERROR_RESPONSES
from app.schemas.ml import (
    EvaluationSummary,
    HorizonMetrics,
    LineageNode,
    MetricOut,
    ModelVersionDetail,
    ModelVersionOut,
    StatusEventOut,
)
from ml.features.schemas import get_schema
from ml.versioning.version_manager import FAMILY_TRAJECTORY

router = APIRouter(prefix="/models", tags=["models"])


Family = Annotated[
    str,
    Query(description="Model family: independent lineages, each with its own champion."),
]


@router.get("", response_model=list[ModelVersionOut], summary="Model versions of one family (base, v1 ... vN)")
async def list_models(session: SessionDep, family: Family = FAMILY_TRAJECTORY) -> list[ModelVersionOut]:
    return [ModelVersionOut.model_validate(m) for m in await queries.model_versions(session, family)]


@router.get("/current", response_model=ModelVersionOut, responses=ERROR_RESPONSES,
            summary="Deployed champion of one model family")
async def current(session: SessionDep, family: Family = FAMILY_TRAJECTORY) -> ModelVersionOut:
    """The champion is whichever version of THIS family is ``deployed`` — never the
    highest number, and never another family's model."""
    mv = await queries.deployed_model(session, family)
    if mv is None:
        raise HTTPException(404, f"no model is deployed for family {family!r}")
    return ModelVersionOut.model_validate(mv)


@router.get("/champions", response_model=dict[str, ModelVersionOut],
            summary="Current champion of every model family")
async def champions(session: SessionDep) -> dict[str, ModelVersionOut]:
    return {f: ModelVersionOut.model_validate(m) for f, m in (await queries.deployed_models(session)).items()}


@router.get("/lineage", response_model=list[LineageNode], summary="Parent/child graph including rejected branches")
async def lineage(session: SessionDep, family: Family = FAMILY_TRAJECTORY) -> list[LineageNode]:
    return [LineageNode.model_validate(m) for m in await queries.model_versions(session, family)]


@router.get("/{version}", response_model=ModelVersionDetail, responses=ERROR_RESPONSES, summary="Model metadata, metrics and status history")
async def get_model(version: str, session: SessionDep) -> ModelVersionDetail:
    mv = await queries.model_version(session, version)
    if mv is None:
        raise HTTPException(404, f"model version {version} not found")
    try:
        feature_schema = get_schema(mv.feature_schema_version).as_dict()
    except ValueError:
        feature_schema = None
    return ModelVersionDetail(
        **ModelVersionOut.model_validate(mv).model_dump(),
        feature_schema=feature_schema,
        metrics=[MetricOut.model_validate(m) for m in await queries.model_metrics(session, version)],
        status_history=[StatusEventOut.model_validate(e) for e in await queries.status_events(session, version)],
        children=await queries.model_children(session, version),
        dataset_manifest=mv.dataset_manifest,
        training_config=mv.training_config,
        test_metrics=mv.test_metrics,
    )


@router.get("/{version}/evaluations", response_model=EvaluationSummary, responses=ERROR_RESPONSES, summary="Operational prediction-vs-actual metrics")
async def model_evaluations(version: str, session: SessionDep, limit: Limit = 50) -> EvaluationSummary:
    if await queries.model_version(session, version) is None:
        raise HTTPException(404, f"model version {version} not found")
    agg = await queries.evaluation_aggregates(session, version)
    by_h = [
        HorizonMetrics(horizon_days=r.forecast_horizon_days, n=r.n, mae_km=r.mae_km, rmse_km=r.rmse_km, median_km=r.median_km, p90_km=r.p90_km)
        for r in agg
    ]
    return EvaluationSummary(
        model_version=version,
        total=sum(h.n for h in by_h),
        by_horizon=by_h,
        recent=[evaluation_out(e) for e in await queries.recent_evaluations(session, version=version, limit=limit)],
    )
