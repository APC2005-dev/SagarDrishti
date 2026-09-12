"""Sea-ice model constants, taken from the training notebook that produced the base ``.pt``.

Source of truth: the notebook cells that built the dataset and the model.
Nothing here is inferred — every value was read from that notebook and, where
observable, verified against the live Copernicus Marine catalogue.

SEA-ICE MODEL INPUT WINDOW = LAST 7 CHRONOLOGICAL DATABASE ENTRIES.
This is an *entry* count, not a calendar-day span. Seven stored observations
form the sequence whatever dates they carry; missing days are never fabricated
to manufacture a 7-calendar-day window. (The trajectory model's 14-entry rule is
the same principle with a different length and is entirely separate from this.)
"""

from __future__ import annotations

# --- official source -------------------------------------------------------
# EUMETSAT OSI SAF AMSR2 L4 sea-ice concentration, Southern Hemisphere, daily.
DATASET_ID = "osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m"
VARIABLE = "ice_conc"          # percent in the source; divided by 100 -> fraction
AUTHORITY = "Copernicus Marine Service (CMEMS) / EUMETSAT OSI SAF"
SOURCE_UNITS = "%"
VALUE_UNITS = "1"              # fraction 0-1 after scaling
NATIVE_RESOLUTION_DEG = 0.1
CRS = "EPSG:4326"              # regular latitude/longitude grid
TEMPORAL_RESOLUTION = "P1D"

# Native grid of the product (verified against the catalogue): 500 x 3600,
# latitude -85.0 .. -35.1, longitude -180.0 .. 179.9.
NATIVE_SHAPE = (500, 3600)
NATIVE_LAT_RANGE = (-85.0, -35.1)
NATIVE_LON_RANGE = (-180.0, 179.9)

# --- preprocessing (notebook) ---------------------------------------------
COARSEN_FACTOR = 5             # .coarsen(latitude=5, longitude=5, boundary="trim").mean(skipna=True)
COARSEN_BOUNDARY = "trim"
GRID_SHAPE = (100, 720)        # 500/5 x 3600/5
GRID_RESOLUTION_DEG = NATIVE_RESOLUTION_DEG * COARSEN_FACTOR  # 0.5 deg

# --- sequence / tensor contract (notebook) --------------------------------
WINDOW = 7                     # entries, NOT calendar days
HORIZONS = (1, 3, 7)           # days ahead, in the model's output channel order
IN_CHANNELS = WINDOW * 2 + 2   # 7 concentration + 7 mask + day-of-year sin + cos = 16
OUT_CHANNELS = len(HORIZONS)   # 3
BASE_CHANNELS = 16             # SmallUNetResidual(base=16)

# Channel layout of the input tensor, in order.
CHANNEL_LAYOUT = (
    tuple(f"ice_conc_entry_{i + 1}" for i in range(WINDOW))
    + tuple(f"valid_mask_entry_{i + 1}" for i in range(WINDOW))
    + ("day_of_year_sin", "day_of_year_cos")
)
# The persistence channel the residual is added to: the LAST concentration entry.
PERSISTENCE_CHANNEL = WINDOW - 1

ARCHITECTURE = "UNetResidual"
ARCHITECTURE_VERSION = "unet_residual_v4"
MODEL_FAMILY = "sea_ice"
BASE_ARTIFACT_FILENAME = "unet_residual_v4_best.pt"
MODEL_FILENAME = "model.pt"

DAYS_PER_YEAR = 365.25         # doy_sin/cos denominator in the notebook
