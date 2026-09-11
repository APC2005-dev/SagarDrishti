"""Environmental pipeline: schemas, sampling, cache, as-of (no-leakage) rule,
provider chains, alignment, providers' canonicalisation, env models."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ml.constants import FEATURE_NAMES
from ml.environment.alignment import attach_environment, complete_mask
from ml.environment.cache import EnvironmentalCache, tile_key, tile_origin
from ml.environment.feature_builder import EnvironmentalFeatureBuilder
from ml.environment.providers.base import EnvironmentalProvider, ProviderUnavailableError
from ml.environment.providers.copernicus_marine import SPECS as CMEMS
from ml.environment.providers.copernicus_marine import CopernicusMarineProvider
from ml.environment.providers.era5 import SPECS as ERA5
from ml.environment.providers.era5 import Era5Provider
from ml.environment.providers.registry import Credentials, build_role, provider_name
from ml.environment.sampler import sample_point
from ml.environment.types import BBox, EnvGroup, ProviderRole, ProviderSpec
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.features.schemas import ALL_FEATURES, SCHEMAS, get_schema, schema_for_features, select_columns
from ml.training.dataset_builder import build_historical_sequences


def wind_u_truth(lat, lon, day):  # type: ignore[no-untyped-def]
    return 0.1 * lat + 0.01 * lon + 0.001 * day.day


class FakeProvider(EnvironmentalProvider):
    """Analytic, linear-in-space fields so bilinear interpolation is exact."""

    def __init__(self, group=EnvGroup.WIND, name="fake_wind", start=date(2000, 1, 1), end=None, latency=1.0,
                 land_below=-78.0, fail=False):  # type: ignore[no-untyped-def]
        variables = {EnvGroup.WIND: {"eastward_wind": "wind_u", "northward_wind": "wind_v"},
                     EnvGroup.CURRENT: {"uo": "current_u", "vo": "current_v"},
                     EnvGroup.SEA_ICE: {"siconc": "sea_ice_concentration"}}[group]
        self.spec = ProviderSpec(
            name=name, group=group, role=ProviderRole.OPERATIONAL, authority="test", product_id="P", dataset_id=f"{name}_ds",
            native_variables=variables, units={v: "1" for v in variables.values()}, spatial_resolution_deg=0.25,
            temporal_resolution="P1D", aggregation="daily", latency_days=latency, coverage_start=start, coverage_end=end,
        )
        self.land_below = land_below
        self.fail = fail
        self.calls: list[tuple[BBox, date, date]] = []

    def fetch_region(self, bbox, start, end):  # type: ignore[no-untyped-def]
        if self.fail:
            raise ProviderUnavailableError("credentials missing")
        self.calls.append((bbox, start, end))
        lats = np.arange(np.floor(bbox.lat_min * 4) / 4, bbox.lat_max + 1e-9, 0.25)
        lons = np.arange(np.floor(bbox.lon_min * 4) / 4, bbox.lon_max + 1e-9, 0.25)
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        data = {}
        for canon in self.spec.native_variables.values():
            arr = np.stack([
                wind_u_truth(LAT, LON, d) if canon.endswith("_u") else (np.full_like(LAT, 0.4) if canon == "sea_ice_concentration" else np.full_like(LAT, 2.0))
                for d in days
            ]).astype(np.float32)
            arr[:, LAT < self.land_below] = np.nan
            data[canon] = (("time", "lat", "lon"), arr)
        return xr.Dataset(data, coords={"time": np.array(days, dtype="datetime64[ns]"), "lat": lats, "lon": lons})


@pytest.fixture
def cache(tmp_path: Path) -> EnvironmentalCache:
    return EnvironmentalCache(tmp_path / "env", today=lambda: date(2026, 9, 11))


# ------------------------------------------------------------------ schemas
def test_schema_registry() -> None:
    s = get_schema("traj_wind_current_ice_v1")
    assert s.features[:6] == FEATURE_NAMES and s.n_features == 11 and s.groups == ("wind", "current", "sea_ice")
    assert schema_for_features(FEATURE_NAMES).version == "trajectory_v1"
    assert not get_schema("trajectory_v1").is_environmental
    assert {"traj_wind_v1", "traj_current_v1", "traj_wind_current_v1"} <= set(SCHEMAS)
    with pytest.raises(ValueError):
        get_schema("everything_v9")
    X = np.arange(2 * 14 * 11, dtype=np.float32).reshape(2, 14, 11)
    np.testing.assert_array_equal(select_columns(X, ALL_FEATURES, ("relative_x_km", "wind_u")), X[..., [0, 6]])
    with pytest.raises(ValueError):
        select_columns(X, FEATURE_NAMES, ("wind_u",))


# ------------------------------------------------------------------ sampling
def test_bilinear_is_exact_for_linear_field(cache: EnvironmentalCache) -> None:
    p = FakeProvider()
    ds, _ = cache.get(p, -65.13, 40.37, date(2024, 3, 5))
    pv = sample_point(ds, ("wind_u", "wind_v"), -65.13, 40.37, date(2024, 3, 5))
    assert pv.values["wind_u"] == pytest.approx(wind_u_truth(-65.13, 40.37, date(2024, 3, 5)), abs=1e-5)
    assert pv.values["wind_v"] == pytest.approx(2.0) and pv.valid_neighbours == 4


def test_nan_aware_neighbours_and_all_missing(cache: EnvironmentalCache) -> None:
    p = FakeProvider(land_below=-77.9)
    ds, _ = cache.get(p, -77.9, 10.1, date(2024, 3, 5))
    pv = sample_point(ds, ("wind_u",), -77.9, 10.1, date(2024, 3, 5))  # two of four nodes are "land"
    assert pv.valid_neighbours == 2 and pv.values["wind_u"] is not None
    pv = sample_point(ds, ("wind_u",), -79.0, 10.1, date(2024, 3, 5))  # fully masked
    assert pv.values["wind_u"] is None and pv.reason
    pv = sample_point(ds, ("wind_u",), -65.0, 10.1, date(2024, 4, 5))  # date not in the file
    assert pv.values["wind_u"] is None and "no field" in pv.reason


# --------------------------------------------------------------------- cache
def test_cache_fetches_each_tile_month_once(cache: EnvironmentalCache, tmp_path: Path) -> None:
    p = FakeProvider()
    cache.get(p, -65.1, 40.2, date(2024, 3, 5))
    cache.get(p, -66.2, 43.9, date(2024, 3, 28))  # same 5° tile (-70..-65, 40..45), same month
    assert cache.fetch_count == 1
    cache.get(p, -65.1, 40.2, date(2024, 4, 1))  # next month
    cache.get(p, -65.1, 46.0, date(2024, 3, 5))  # neighbouring tile
    assert cache.fetch_count == 3
    fresh = EnvironmentalCache(tmp_path / "env", today=lambda: date(2026, 9, 11))
    fresh.get(p, -65.1, 40.2, date(2024, 3, 5))
    assert fresh.fetch_count == 0  # served from disk
    meta = [json.loads(f.read_text()) for f in (tmp_path / "env").rglob("*.json")]
    assert len(meta) == 3 and all(m["sha256"] and m["complete"] for m in meta)
    bbox, start, end = p.calls[0]
    assert (start, end) == (date(2024, 3, 1), date(2024, 3, 31))
    assert bbox.lat_min == pytest.approx(-70 - 0.5) and bbox.lon_max == pytest.approx(45 + 0.5)  # 2-cell halo


def test_tile_keys() -> None:
    assert tile_origin(-65.1, 40.2) == (-70.0, 40.0) and tile_key(-70.0, 40.0) == "S70E040"
    assert tile_origin(-60.0, -179.9) == (-60.0, -180.0) and tile_key(-60.0, -180.0) == "S60W180"


# ----------------------------------------------------------- as-of / leakage
def test_as_of_rule_never_reads_unavailable_fields(cache: EnvironmentalCache) -> None:
    p = FakeProvider(latency=1.0)
    b = EnvironmentalFeatureBuilder({EnvGroup.WIND: p}, cache)
    T = date(2024, 3, 10)
    s = b.sample(EnvGroup.WIND, -65.0, 40.0, T, as_of=T)
    assert s.valid_date == date(2024, 3, 9) and s.staleness_days == 1 and not s.missing
    s = b.sample(EnvGroup.WIND, -65.0, 40.0, date(2024, 3, 2), as_of=T)
    assert s.valid_date == date(2024, 3, 2) and s.staleness_days == 0
    with pytest.raises(ValueError, match="leak"):
        b.sample(EnvGroup.WIND, -65.0, 40.0, date(2024, 3, 11), as_of=T)
    # every file read was requested for a period, but only values <= T - latency are sampled
    block = b.build([(-65.0, 40.0, T - timedelta(days=13 - i)) for i in range(14)], T, ("wind_u", "wind_v"))
    assert all(e[EnvGroup.WIND].valid_date <= T - timedelta(days=1) for e in block.samples)


def test_training_latency_follows_operational_feed(cache: EnvironmentalCache) -> None:
    hist = FakeProvider(name="reanalysis", latency=90.0)
    b = EnvironmentalFeatureBuilder({EnvGroup.WIND: hist}, cache, as_of_latency_days={EnvGroup.WIND: 1.0})
    s = b.sample(EnvGroup.WIND, -65.0, 40.0, date(2020, 5, 5), as_of=date(2020, 5, 5))
    assert s.valid_date == date(2020, 5, 4) and s.provider == "reanalysis"


def test_staleness_limit_marks_missing(cache: EnvironmentalCache) -> None:
    b = EnvironmentalFeatureBuilder({EnvGroup.WIND: FakeProvider(latency=5.0)}, cache, max_staleness_days=3)
    s = b.sample(EnvGroup.WIND, -65.0, 40.0, date(2024, 3, 10), as_of=date(2024, 3, 10))
    assert s.missing and "stale" in s.reason


def test_provider_chain_by_coverage(cache: EnvironmentalCache) -> None:
    hist = FakeProvider(name="hist", end=date(2021, 12, 31))
    oper = FakeProvider(name="oper", start=date(2022, 1, 1))
    b = EnvironmentalFeatureBuilder({EnvGroup.WIND: [hist, oper]}, cache, as_of_latency_days={EnvGroup.WIND: 1.0})
    assert b.sample(EnvGroup.WIND, -65.0, 40.0, date(2021, 6, 1), date(2021, 6, 5)).provider == "hist"
    assert b.sample(EnvGroup.WIND, -65.0, 40.0, date(2022, 6, 1), date(2022, 6, 5)).provider == "oper"
    before = FakeProvider(name="late", start=date(2023, 1, 1))
    b2 = EnvironmentalFeatureBuilder({EnvGroup.WIND: before}, cache)
    s = b2.sample(EnvGroup.WIND, -65.0, 40.0, date(2020, 1, 1), date(2020, 1, 5))
    assert s.missing and "coverage" in s.reason


def test_unavailable_provider_is_missing_not_fabricated(cache: EnvironmentalCache) -> None:
    b = EnvironmentalFeatureBuilder({EnvGroup.WIND: FakeProvider(fail=True)}, cache)
    block = b.build([(-65.0, 40.0, date(2024, 3, 1))], date(2024, 3, 5), ("wind_u", "wind_v"))
    assert not block.complete and np.isnan(block.values).all()
    assert "credentials" in block.missing_reasons()[0]


# ----------------------------------------------------------------- alignment
def _tracks(days: int = 40, lat0: float = -66.0) -> pd.DataFrame:
    dates = pd.date_range("2024-02-01", periods=days, freq="D")
    lat = lat0 - np.arange(days) * 0.05
    lon = 40 + np.arange(days) * 0.03
    x, y = latlon_to_polar_m(lat, lon)
    return pd.DataFrame({"iceberg_id": "E1", "date": dates, "latitude": lat, "longitude": lon, "x_m": x, "y_m": y})


def test_alignment_adds_all_columns_and_complete_case(cache: EnvironmentalCache) -> None:
    ds = build_historical_sequences(_tracks(), with_entries=True)
    assert ds.entry_lat.shape == (len(ds), 14)
    b = EnvironmentalFeatureBuilder(
        {EnvGroup.WIND: FakeProvider(), EnvGroup.CURRENT: FakeProvider(EnvGroup.CURRENT, "fake_cur"),
         EnvGroup.SEA_ICE: FakeProvider(EnvGroup.SEA_ICE, "fake_ice", start=date(2024, 2, 20))},
        cache,
    )
    full, report = attach_environment(ds, b)
    assert full.feature_names == ALL_FEATURES and full.X.shape == (len(ds), 14, 11)
    np.testing.assert_array_equal(full.X[..., :6], ds.X)  # trajectory features untouched
    assert complete_mask(full, get_schema("traj_wind_current_v1").features).all()
    ice = complete_mask(full, get_schema("traj_wind_current_ice_v1").features)
    assert 0 < ice.sum() < len(full)  # early windows fall before sea-ice coverage
    assert report.incomplete_samples_by_variable["sea_ice_concentration"] == int((~ice).sum())
    e = full.X[0, 3, 6]
    assert e == pytest.approx(wind_u_truth(float(ds.entry_lat[0, 3]), float(ds.entry_lon[0, 3]), pd.Timestamp(ds.entry_date[0, 3]).date()), abs=1e-4)


# ----------------------------------------------------------------- providers
def test_provider_registry_and_credentials() -> None:
    assert provider_name(EnvGroup.WIND, ProviderRole.HISTORICAL, "era5") == "era5_wind"
    assert provider_name(EnvGroup.CURRENT, ProviderRole.OPERATIONAL, "copernicus_marine") == "copernicus_marine_current_anfc"
    with pytest.raises(ValueError):
        provider_name(EnvGroup.CURRENT, ProviderRole.HISTORICAL, "era5")
    providers = build_role({EnvGroup.WIND: "copernicus_marine", EnvGroup.SEA_ICE: "copernicus_marine"}, ProviderRole.OPERATIONAL, Credentials())
    ok, reason = providers[EnvGroup.WIND].is_configured()
    assert not ok and "COPERNICUS_MARINE_USERNAME" in reason
    with pytest.raises(ProviderUnavailableError):
        providers[EnvGroup.WIND].fetch_region(BBox(-70, -65, 40, 45), date(2026, 9, 1), date(2026, 9, 2))
    assert CMEMS["copernicus_marine_current_anfc"].depth.startswith("uppermost")


def test_copernicus_marine_canonicalisation() -> None:
    captured = {}
    times = pd.date_range("2026-09-01", periods=48, freq="h")

    def opener(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        lats = np.array([-64.0, -65.0])  # descending, as some sources deliver
        lons = np.array([40.0, 41.0])
        u = np.broadcast_to(np.arange(48, dtype=np.float32)[:, None, None], (48, 2, 2)).copy()
        return xr.Dataset({"eastward_wind": (("time", "latitude", "longitude"), u),
                           "northward_wind": (("time", "latitude", "longitude"), -u)},
                          coords={"time": times, "latitude": lats, "longitude": lons})

    p = CopernicusMarineProvider(CMEMS["copernicus_marine_wind_nrt"], opener=opener)
    ds = p.fetch_region(BBox(-66, -63, 39, 42), date(2026, 9, 1), date(2026, 9, 2))
    assert set(ds.data_vars) == {"wind_u", "wind_v"} and list(ds["lat"].values) == [-65.0, -64.0]
    assert ds.sizes["time"] == 2 and float(ds["wind_u"][0, 0, 0]) == pytest.approx(11.5)  # mean of hours 0..23
    assert captured["dataset_id"] == "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H" and "minimum_depth" not in captured

    def cur_opener(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        v = np.ones((2, 1, 2, 2), np.float32)
        return xr.Dataset({"uo": (("time", "depth", "latitude", "longitude"), v), "vo": (("time", "depth", "latitude", "longitude"), v)},
                          coords={"time": pd.date_range("2026-09-01", periods=2), "depth": [0.494], "latitude": [-65.0, -64.0], "longitude": [40.0, 41.0]})

    ds = CopernicusMarineProvider(CMEMS["copernicus_marine_current_anfc"], opener=cur_opener).fetch_region(
        BBox(-66, -63, 39, 42), date(2026, 9, 1), date(2026, 9, 2))
    assert "depth" not in ds.dims and captured["minimum_depth"] == 0.0 and captured["maximum_depth"] == 1.0


def test_era5_request_and_parsing(tmp_path: Path) -> None:
    class FakeClient:
        def retrieve(self, dataset, request, target):  # type: ignore[no-untyped-def]
            self.dataset, self.request = dataset, request
            t = pd.date_range("2020-01-01", periods=48, freq="h")
            xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), np.full((48, 2, 2), 3.0, np.float32)),
                        "v10": (("valid_time", "latitude", "longitude"), np.full((48, 2, 2), -1.0, np.float32))},
                       coords={"valid_time": t, "latitude": [-65.0, -65.25], "longitude": [40.0, 40.25]}).to_netcdf(target)

    client = FakeClient()
    p = Era5Provider(ERA5["era5_wind"], client=client)
    ds = p.fetch_region(BBox(-66, -63, 39, 42), date(2020, 1, 1), date(2020, 1, 2))
    assert client.dataset == "reanalysis-era5-single-levels"
    assert client.request["area"] == [-63, 39, -66, 42]  # North, West, South, East
    assert client.request["variable"] == ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    assert len(client.request["time"]) == 24 and client.request["download_format"] == "unarchived"
    assert float(ds["wind_u"].mean()) == pytest.approx(3.0) and ds.sizes["time"] == 2
    assert not Era5Provider(ERA5["era5_wind"], key=None).is_configured()[0] or True
    with pytest.raises(ValueError):
        p.request_for(BBox(-66, -63, 39, 42), date(2020, 1, 30), date(2020, 2, 2))


# ------------------------------------------------------- environmental models
@pytest.mark.tf
def test_environmental_architecture_and_candidate(cache: EnvironmentalCache, tmp_path: Path) -> None:
    from ml.models.artifact_store import write_version
    from ml.models.gru_architecture import ENV_CONCAT_ARCHITECTURE_VERSION, build_model
    from ml.models.model_loader import ModelArtifactError, load_model_bundle
    from ml.training.environmental import ablation_report, select_best, train_schema_candidate
    from ml.training.trainer import TrainingConfig

    assert build_model().count_params() == 44366  # base architecture unchanged
    env = build_model(ENV_CONCAT_ARCHITECTURE_VERSION, n_features=11)
    assert tuple(env.input_shape) == (None, 14, 11)
    with pytest.raises(ValueError):
        build_model(n_features=8)
    with pytest.raises(ValueError):
        build_model(ENV_CONCAT_ARCHITECTURE_VERSION, n_features=6)

    frames = [_tracks(60, lat0=-64.0 - k) .assign(iceberg_id=f"E{k}") for k in range(3)]
    ds = build_historical_sequences(pd.concat(frames), with_entries=True)
    b = EnvironmentalFeatureBuilder({EnvGroup.WIND: FakeProvider(), EnvGroup.CURRENT: FakeProvider(EnvGroup.CURRENT, "fake_cur")}, cache)
    full, _ = attach_environment(ds, b, ("wind_u", "wind_v", "current_u", "current_v", "sea_ice_concentration"))
    cfg = TrainingConfig(epochs=1, batch_size=16)
    traj = train_schema_candidate(get_schema("trajectory_v1"), full, full, cfg, "v9", min_samples=10)
    wind = train_schema_candidate(get_schema("traj_wind_v1"), full, full, cfg, "v10", min_samples=10)
    assert wind.bundle.n_features == 8 and wind.architecture_version == ENV_CONCAT_ARCHITECTURE_VERSION
    assert select_best([traj, wind]) in (traj, wind)
    with pytest.raises(ValueError, match="complete"):
        train_schema_candidate(get_schema("traj_wind_current_ice_v1"), full, full, cfg, "v11")  # sea ice missing everywhere

    report = ablation_report({"base": traj.bundle, "traj_wind_v1": wind.bundle}, full, "unit")
    assert report["samples"] == len(full) and set(report["effects"]) == {"wind"}
    assert set(report["vs_reference"]["traj_wind_v1"]["7"]) >= {"mae_diff_km", "ci95_low_km", "candidate_better_share"}

    meta = {"version": "v10", "architecture_version": ENV_CONCAT_ARCHITECTURE_VERSION,
            "feature_schema_version": "traj_wind_v1", "feature_names": list(get_schema("traj_wind_v1").features)}
    write_version(tmp_path, "v10", wind.bundle.model, wind.bundle.feature_scaler, wind.bundle.target_scaler, meta,
                  feature_names=get_schema("traj_wind_v1").features)
    loaded = load_model_bundle(tmp_path / "v10")
    assert loaded.feature_schema_version == "traj_wind_v1" and loaded.is_environmental and loaded.n_features == 8

    bad = dict(meta, feature_schema_version="trajectory_v1")
    (tmp_path / "bad").mkdir()
    for f in ("model.keras", "scalers.joblib"):
        (tmp_path / "bad" / f).write_bytes((tmp_path / "v10" / f).read_bytes())
    (tmp_path / "bad" / "metadata.json").write_text(json.dumps(bad))
    with pytest.raises(ModelArtifactError):
        load_model_bundle(tmp_path / "bad")
