"""Application settings. Every secret and environment-specific value comes from
environment variables (see ``.env.example``); nothing sensitive has a default."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Repo-root .env first, then CWD .env (later files take precedence); real env vars override both.
    model_config = SettingsConfigDict(env_file=(str(REPO_ROOT / ".env"), ".env"), env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "SAGAR DRISHTI"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    log_json: bool = True

    # --- infrastructure (no credential defaults) -------------------------------
    database_url: SecretStr = Field(..., description="postgresql+psycopg://user:pass@host:5432/db")
    redis_url: SecretStr = Field(SecretStr("redis://localhost:6379/0"))
    secret_key: SecretStr | None = None
    # NoDecode: accept "a,b" from env files instead of requiring a JSON list.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]

    # --- USNIC source ----------------------------------------------------------
    usnic_csv_url: str = "https://usicecenter.gov/File/DownloadCurrent?pId=134"
    usnic_product_page_url: str = "https://usicecenter.gov/Products/AntarcIcebergs"
    usnic_timeout_seconds: float = 30.0
    usnic_max_bytes: int = 5_000_000
    usnic_user_agent: str = "SagarDrishti/1.0 (Antarctic iceberg monitoring)"
    usnic_retries: int = 2
    ingestion_interval_hours: float = 72.0
    # An official observation older than this is flagged stale in the API/UI.
    stale_after_days: int = 10
    # A feed whose last successful fetch is older than this is DEGRADED.
    feed_degraded_after_hours: float = 96.0

    # --- basemap imagery (NASA GIBS, native EPSG:3031) — proxied and cached here,
    # so the browser only ever talks to this API ---------------------------------
    basemap_enabled: bool = True
    basemap_url_template: str = (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3031/best/{layer}/default/{tile_matrix_set}/{z}/{row}/{col}.jpeg"
    )
    basemap_cache_dir: Path = REPO_ROOT / "data" / "processed" / "basemap"
    basemap_timeout_seconds: float = 20.0

    # --- environmental features (environmental model versions only; base/v1 never use them)
    env_enabled: bool = True
    env_wind_provider: str = "copernicus_marine"
    env_current_provider: str = "copernicus_marine"
    env_sea_ice_provider: str = "copernicus_marine"
    env_historical_wind_provider: str = "era5"
    env_historical_current_provider: str = "copernicus_marine"
    env_historical_sea_ice_provider: str = "copernicus_marine"
    env_cache_dir: Path = REPO_ROOT / "data" / "processed" / "environment"
    env_max_staleness_days: int = 3
    env_candidate_schemas: Annotated[list[str], NoDecode] = [
        "trajectory_v1", "traj_wind_v1", "traj_current_v1", "traj_wind_current_v1", "traj_wind_current_ice_v1",
    ]
    env_training_max_historical_samples: int = 30_000
    env_training_max_test_samples: int = 8_000
    env_archive_forecast_fields: bool = True
    env_overlay_resolution_deg: float = 1.0
    # Trajectory-only model used when an environmental champion lacks inputs for an iceberg.
    # Empty = nearest trajectory-only ancestor of the champion.
    env_fallback_model_version: str | None = None
    copernicus_marine_username: SecretStr | None = Field(
        None, validation_alias=AliasChoices("COPERNICUS_MARINE_USERNAME", "COPERNICUSMARINE_SERVICE_USERNAME")
    )
    copernicus_marine_password: SecretStr | None = Field(
        None, validation_alias=AliasChoices("COPERNICUS_MARINE_PASSWORD", "COPERNICUSMARINE_SERVICE_PASSWORD")
    )
    cds_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("CDSAPI_KEY", "CDS_API_KEY"))
    cds_api_url: str = Field("https://cds.climate.copernicus.eu/api", validation_alias=AliasChoices("CDSAPI_URL", "CDS_API_URL"))

    # --- paths -----------------------------------------------------------------
    raw_data_dir: Path = REPO_ROOT / "data" / "raw"
    models_dir: Path = REPO_ROOT / "models"
    bootstrap_dataset_path: Path = REPO_ROOT / "data" / "bootstrap" / "consolidated_database_v8.0.zip"

    # --- sequence adapter ------------------------------------------------------
    adapter_strategy: str = "bootstrap_v1"
    adapter_max_gap_days: float = 730.0

    # --- retraining policy -----------------------------------------------------
    retrain_check_interval_hours: float = 168.0
    retrain_min_new_evaluations: int = 50
    retrain_min_new_icebergs: int = 5
    retrain_min_days_since_last_training: float = 7.0
    retrain_strategy: str = "fine_tune"
    retrain_epochs: int = 20
    retrain_learning_rate: float = 1e-4
    retrain_historical_replay_samples: int = 50_000
    retrain_operational_validation_fraction: float = 0.15
    retrain_operational_test_fraction: float = 0.15
    # Fixed historical protocol boundaries = the base model's split (notebook §4).
    historical_train_end: date = date(2018, 8, 23)
    historical_validation_end: date = date(2022, 3, 9)

    # --- sea-ice model (independent core model, own lineage) -------------------
    # The sea-ice model is a first-class core model, NOT an environmental feature.
    # Its input window is 7 chronological database entries (see ml.seaice.constants).
    seaice_enabled: bool = True
    seaice_dataset_id: str = "osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m"
    seaice_data_dir: Path = REPO_ROOT / "data" / "processed" / "seaice"
    # The scheduler CHECKS this often; it does not mean a new entry exists each
    # time, and it never by itself creates a model version.
    seaice_poll_interval_hours: float = 72.0
    seaice_initial_backfill_days: int = 120
    seaice_max_entries_per_run: int = 40
    # Historical backfill (``load-historical``). The default start matches the
    # period the base model was trained on (notebook: 2024-09-01 .. 2026-08-25);
    # the source itself reaches further back, so lower this to load more.
    seaice_historical_start: date = date(2024, 9, 1)
    seaice_historical_max_entries: int | None = None  # None = no cap for a one-off backfill
    # Contiguous days fetched per read. One-day-at-a-time costs a round trip per
    # day (~16s); a chunked slice amortises it (~0.65s/day measured).
    seaice_fetch_chunk_size: int = 20
    # Retraining eligibility.
    seaice_retrain_min_new_observations: int = 30
    seaice_retrain_min_days_since_last_training: float = 30.0
    seaice_retrain_min_evaluations: int = 10
    # Performance-degradation trigger, measured on stored operational evaluations.
    seaice_degradation_horizon_days: int = 7
    seaice_degradation_max_rmse: float | None = None
    seaice_degradation_relative_tolerance: float | None = 0.20
    seaice_degradation_min_samples: int = 10
    # Candidate training.
    seaice_retrain_epochs: int = 30
    seaice_retrain_learning_rate: float = 1e-4
    seaice_retrain_batch_size: int = 8
    seaice_retrain_patience: int = 5
    seaice_retrain_validation_fraction: float = 0.2
    seaice_retrain_seed: int = 42
    # Promotion policy: a candidate must beat the champion by this margin on the
    # primary horizon, and must not regress the short horizon beyond the cap.
    seaice_promotion_primary_horizon: int = 7
    seaice_promotion_min_relative_improvement: float = 0.02
    seaice_promotion_max_short_horizon_regression: float = 0.05
    seaice_promotion_min_evaluation_samples: int = 5

    promotion_primary_horizon: int = 7
    promotion_min_relative_improvement: float = 0.02
    promotion_max_short_horizon_regression: float = 0.05
    promotion_min_evaluation_samples: int = 30

    @field_validator("cors_origins", "env_candidate_schemas", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def sync_database_url(self) -> str:
        return self.database_url.get_secret_value().replace("postgresql+asyncpg://", "postgresql+psycopg://")

    @property
    def async_database_url(self) -> str:
        url = self.database_url.get_secret_value()
        for prefix in ("postgresql+psycopg://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix):]
        return url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
