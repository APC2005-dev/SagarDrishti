from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from ml.features.coordinate_transform import haversine_km, latlon_to_polar_m, polar_m_to_latlon
from ml.features.seasonal_features import season_features
from ml.features.trajectory_features import (
    FeatureConstructionError,
    build_feature_matrix,
    build_target_vector,
    elapsed_days_between,
)


class TestCoordinateTransform:
    def test_matches_notebook_epsg3031_values(self) -> None:
        # pyproj output recorded in the original notebook run, iceberg A01 on 1978-10-22.
        x, y = latlon_to_polar_m([-56.0], [-34.2])
        assert x[0] == pytest.approx(-2.137135e06, rel=1e-6)
        assert y[0] == pytest.approx(3.144699e06, rel=1e-6)

    def test_round_trip(self) -> None:
        lats = np.array([-53.94, -71.19, -66.07, -89.5])
        lons = np.array([-26.43, -101.65, 143.15, 179.9])
        x, y = latlon_to_polar_m(lats, lons)
        lat2, lon2 = polar_m_to_latlon(x, y)
        np.testing.assert_allclose(lat2, lats, atol=1e-9)
        np.testing.assert_allclose(lon2, lons, atol=1e-9)

    def test_haversine(self) -> None:
        # One degree of latitude ~ 111.2 km on the notebook's sphere.
        assert haversine_km(-60.0, 10.0, -61.0, 10.0) == pytest.approx(111.195, abs=0.01)
        assert haversine_km(-60.0, 10.0, -60.0, 10.0) == pytest.approx(0.0)

    def test_antimeridian_distance_is_short(self) -> None:
        assert haversine_km(-67.0, 179.9, -67.0, -179.9) < 10


class TestSeasonal:
    def test_day_of_year_encoding(self) -> None:
        s, c = season_features([date(2026, 1, 1)])
        angle = 2 * np.pi * 1 / 365.25
        assert s[0] == pytest.approx(np.sin(angle))
        assert c[0] == pytest.approx(np.cos(angle))


def _line(n: int, step_days: int) -> tuple[np.ndarray, np.ndarray, list[date]]:
    x = np.arange(n, dtype=float) * 7000.0  # 7 km per entry in x
    y = np.arange(n, dtype=float) * -3500.0
    dates = [date(2026, 3, 1) + timedelta(days=i * step_days) for i in range(n)]
    return x, y, dates


class TestFeatureMatrix:
    def test_daily_sequence_matches_notebook_formula(self) -> None:
        x, y, dates = _line(14, 1)
        fm = build_feature_matrix(x, y, dates)
        assert fm.features.shape == (14, 6)
        np.testing.assert_allclose(fm.features[:, 0], (x - x[-1]) / 1000.0, rtol=1e-6)
        np.testing.assert_allclose(fm.features[1:, 2], np.diff(x) / 1000.0, rtol=1e-6)
        assert fm.features[0, 2] == fm.features[1, 2]  # first velocity copied
        assert fm.anchor_x_m == x[-1]

    def test_weekly_velocity_uses_elapsed_days_not_entry_count(self) -> None:
        x, y, dates = _line(14, 7)
        fm = build_feature_matrix(x, y, dates)
        # 7 km moved over 7 days = 1 km/day, not 7 km/entry.
        np.testing.assert_allclose(fm.features[1:, 2], 1.0, rtol=1e-6)
        np.testing.assert_allclose(fm.features[1:, 3], -0.5, rtol=1e-6)
        np.testing.assert_array_equal(fm.elapsed_days[1:], 7.0)

    def test_irregular_gaps(self) -> None:
        x, y, _ = _line(14, 1)
        gaps = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 130]  # BYU -> USNIC seam
        dates = [date(2026, 1, 1)]
        for g in gaps:
            dates.append(dates[-1] + timedelta(days=g))
        fm = build_feature_matrix(x, y, dates)
        assert fm.features[-1, 2] == pytest.approx(7.0 / 130.0, rel=1e-5)

    def test_rejects_non_increasing_dates(self) -> None:
        x, y, dates = _line(14, 1)
        dates[5] = dates[4]
        with pytest.raises(FeatureConstructionError):
            build_feature_matrix(x, y, dates)

    def test_rejects_wrong_length(self) -> None:
        x, y, dates = _line(13, 1)
        with pytest.raises(FeatureConstructionError):
            build_feature_matrix(x, y, dates)

    def test_rejects_non_finite(self) -> None:
        x, y, dates = _line(14, 1)
        x[3] = np.nan
        with pytest.raises(FeatureConstructionError):
            build_feature_matrix(x, y, dates)

    def test_elapsed_days(self) -> None:
        d = [date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 9)]
        np.testing.assert_array_equal(elapsed_days_between(d), [0, 7, 1])


class TestTargets:
    def test_partial_targets_are_nan(self) -> None:
        t = build_target_vector(0.0, 0.0, [None] * 6 + [7000.0], [None] * 6 + [-2000.0])
        assert t.shape == (14,)
        assert np.isnan(t[:12]).all()
        assert t[12] == pytest.approx(7.0)
        assert t[13] == pytest.approx(-2.0)
