"""Database-enforced permanence: forecasts/evaluations are append-only, observations are never deleted."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.models.ml import Forecast, ForecastRun, ForecastSet, ModelVersion
from app.models.tracking import Iceberg, Observation

pytestmark = pytest.mark.db


@pytest.fixture
def seeded_forecast(db):  # type: ignore[no-untyped-def]
    db.add(Iceberg(iceberg_id="A81", status="active"))
    db.flush()
    obs = Observation(iceberg_id="A81", observation_date=date(2026, 9, 10), latitude=-57.36, longitude=-47.22,
                      geom="SRID=4326;POINT(-47.22 -57.36)", source="usnic", provenance="official_usnic", content_hash="h", revision=1)
    mv = ModelVersion(version="v1", version_number=1, architecture="GRU", architecture_version="gru_entry14_to_day7_v1",
                      input_sequence_length=14, input_semantics="chronological_observation_entries", forecast_horizon_days=7,
                      feature_names=[], model_path="m", scaler_path="s", metadata_path="x", artifact_sha256={"model.keras": "a"},
                      artifact_origin="test", status="deployed")
    db.add_all([obs, mv])
    db.flush()
    run = ForecastRun(model_version="v1", trigger="test", status="success")
    db.add(run)
    db.flush()
    fs = ForecastSet(forecast_run_id=run.id, iceberg_id="A81", model_version="v1", generated_at=datetime.now(UTC),
                     anchor_observation_id=obs.id, latest_observation_date=obs.observation_date, adapter_version="t",
                     adapter_strategy="bootstrap_v1", input_observation_ids=[obs.id], input_entries=[], input_features=[], diagnostics={})
    db.add(fs)
    db.flush()
    db.add(Forecast(forecast_set_id=fs.id, iceberg_id="A81", model_version="v1", generated_at=fs.generated_at,
                    latest_observation_date=obs.observation_date, forecast_horizon_days=1, forecast_date=date(2026, 9, 11),
                    predicted_latitude=-57.4, predicted_longitude=-47.2, predicted_x_m=0, predicted_y_m=0,
                    geom="SRID=4326;POINT(-47.2 -57.4)"))
    db.commit()
    return db


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE ml.forecasts SET predicted_latitude = -60",
        "DELETE FROM ml.forecasts",
        "UPDATE ml.forecast_sets SET status = 'x'",
        "DELETE FROM tracking.observations",
        "DELETE FROM ml.model_versions",
        "UPDATE ml.model_versions SET artifact_sha256 = '{}'::jsonb",
        "UPDATE ml.model_versions SET parent_version = NULL, version = 'v9'",
    ],
)
def test_permanence_enforced(seeded_forecast, sql: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DBAPIError, match="append-only|immutable"):
        seeded_forecast.execute(text(sql))
    seeded_forecast.rollback()


def test_status_changes_are_allowed(seeded_forecast) -> None:  # type: ignore[no-untyped-def]
    seeded_forecast.execute(text("UPDATE ml.model_versions SET status = 'archived', status_reason = 'test'"))
    seeded_forecast.execute(text("UPDATE tracking.observations SET revision = 2, latitude = -57.5"))  # source correction path
    seeded_forecast.commit()


def test_single_deployed_model(seeded_forecast) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy.exc import IntegrityError

    seeded_forecast.add(ModelVersion(version="v2", version_number=2, architecture="GRU", architecture_version="gru_entry14_to_day7_v1",
                                     input_sequence_length=14, input_semantics="x", forecast_horizon_days=7, feature_names=[],
                                     model_path="m", scaler_path="s", metadata_path="x", artifact_sha256={}, artifact_origin="t",
                                     status="deployed"))
    with pytest.raises(IntegrityError):
        seeded_forecast.commit()
    seeded_forecast.rollback()
