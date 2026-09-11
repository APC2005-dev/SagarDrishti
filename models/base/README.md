# models/base — base research artifact

This directory must contain the files exported by section 8 of
`notebooks/01_gru_iceberg_trajectory_model.ipynb`:

| file | produced by |
| --- | --- |
| `global_gru_trajectory_model.keras` | `model.save(...)` |
| `global_gru_trajectory_scalers.joblib` | `joblib.dump({feature_scaler, target_scaler, history_days, forecast_days, feature_names})` |
| `metadata.json` | written by the export step (the tracked copy describes the original September 2026 run) |

The binaries of the original run were distributed as `person3_global_gru_results.zip`
from the Colab session and are **not** in this repository. Either copy the two
binaries from that zip here (keep the tracked `metadata.json`), or re-run the
notebook on Colab and copy all three exported files (a re-run is a new training
run; its own `metadata.json` records its exact metrics and all seven p90 radii).
Then:

```bash
dev bootstrap        # registers base (validates shapes + records sha256), creates and deploys v1
```

`dev bootstrap` refuses to continue if the files are missing or violate the
tensor contract (input `(None, 14, 6)`, output `(None, 14)`, 6-feature / 14-target
scalers). Until then the platform runs with official positions only and shows
"model unavailable" — it never substitutes a different model.

`make reproduce-base` (the same recipe as a script) writes to `models/base-reproduced/`
instead; promoting that to `base` is a deliberate human decision.

Files in `models/*/` are immutable once registered: never edit or overwrite them.
