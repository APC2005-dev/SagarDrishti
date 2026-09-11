"""Model-contract constants shared by every module.

These values define the tensor contract of architecture ``gru_entry14_to_day7_v1``
and must match the base artifact (notebook §3–§5). They are not tuning
knobs: changing them means creating a new architecture version.
"""

from __future__ import annotations

SEQUENCE_LENGTH = 14
FORECAST_DAYS = 7
OUTPUT_SIZE = FORECAST_DAYS * 2  # Day1 X, Day1 Y, ..., Day7 X, Day7 Y

# Exact names stored in global_gru_trajectory_scalers.joblib (notebook §3).
FEATURE_NAMES: tuple[str, ...] = (
    "relative_x_km",
    "relative_y_km",
    "vx_km_per_day",
    "vy_km_per_day",
    "season_sin",
    "season_cos",
)
N_FEATURES = len(FEATURE_NAMES)

ARCHITECTURE = "GRU"
ARCHITECTURE_VERSION = "gru_entry14_to_day7_v1"

# Research semantics (base model) vs production semantics (v1+).
INPUT_SEMANTICS_RESEARCH = "consecutive_daily_positions"
INPUT_SEMANTICS_PRODUCTION = "chronological_observation_entries"

# Days per year used by the notebook for the seasonal cycle.
DAYS_PER_YEAR = 365.25

# Mean Earth radius used by the notebook's haversine (§6).
EARTH_RADIUS_KM = 6371.0088

GEOGRAPHIC_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3031"
