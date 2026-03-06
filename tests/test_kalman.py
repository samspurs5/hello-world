"""Tests for VariableNoiseKalmanFilter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalmangps import GPSConfig, GPSMeasurement, VariableNoiseKalmanFilter
from kalmangps.filters import KalmanFilterResult
from kalmangps.sensors.gps import latlon_to_xy, xy_to_latlon
from tests.conftest import make_gps_df


def _make_measurements(df: pd.DataFrame) -> tuple[list[GPSMeasurement], GPSConfig]:
    cfg = GPSConfig(lat_ref=None, lon_ref=None)
    measurements = cfg.measurements_from_df(df, auto_ref=True)
    return measurements, cfg


class TestGPSCoordinateConversion:

    def test_round_trip(self):
        lat, lon = 51.5, -0.1
        x, y = latlon_to_xy(lat, lon, lat, lon)
        assert abs(x) < 1e-6
        assert abs(y) < 1e-6

    def test_inverse(self):
        lat_ref, lon_ref = 48.8566, 2.3522
        lat_in, lon_in = 48.86, 2.36
        x, y = latlon_to_xy(lat_in, lon_in, lat_ref, lon_ref)
        lat_out, lon_out = xy_to_latlon(x, y, lat_ref, lon_ref)
        assert abs(lat_out - lat_in) < 1e-8
        assert abs(lon_out - lon_in) < 1e-8


class TestVariableNoiseKalmanFilter:

    def test_output_shape(self, simple_df):
        measurements, cfg = _make_measurements(simple_df)
        kf = VariableNoiseKalmanFilter()
        result = kf.filter(measurements, cfg)

        n = len(measurements)
        assert result.state_means.shape == (n, 4)
        assert result.state_covariances.shape == (n, 4, 4)
        assert result.innovations.shape == (n, 2)
        assert len(result.timestamps) == n

    def test_position_xy_shape(self, simple_df):
        measurements, cfg = _make_measurements(simple_df)
        result = VariableNoiseKalmanFilter().filter(measurements, cfg)
        assert result.position_xy().shape == (len(measurements), 2)

    def test_velocity_xy_shape(self, simple_df):
        measurements, cfg = _make_measurements(simple_df)
        result = VariableNoiseKalmanFilter().filter(measurements, cfg)
        assert result.velocity_xy().shape == (len(measurements), 2)

    def test_position_uncertainty_positive(self, simple_df):
        measurements, cfg = _make_measurements(simple_df)
        result = VariableNoiseKalmanFilter().filter(measurements, cfg)
        unc = result.position_uncertainty()
        assert (unc > 0).all()

    def test_covariance_uncertainty_decreases_over_time(self, simple_df):
        """Filter should gain confidence: uncertainty at end < start."""
        measurements, cfg = _make_measurements(simple_df)
        result = VariableNoiseKalmanFilter().filter(measurements, cfg)
        unc = result.position_uncertainty()
        # After many consistent observations, uncertainty should generally be
        # lower than initial (which starts at 1e4).
        assert unc[-1, 0] < unc[0, 0]

    def test_filtered_position_close_to_raw(self, simple_df):
        """Filtered lat/lon should be within ~50 m of raw observations."""
        measurements, cfg = _make_measurements(simple_df)
        result = VariableNoiseKalmanFilter().filter(measurements, cfg)

        raw_xy = np.stack([m.values for m in measurements])
        filtered_xy = result.position_xy()
        # Euclidean distance per point in metres
        dists = np.linalg.norm(filtered_xy - raw_xy, axis=1)
        assert np.mean(dists) < 50.0

    def test_variable_noise_matters(self):
        """High-accuracy fix should pull the estimate more than a low-accuracy one."""
        rng = np.random.default_rng(99)
        df = make_gps_df(n=10, accuracy=5.0, rng=rng)
        measurements_low_noise, cfg = _make_measurements(df)

        # Override: make all observations very noisy
        high_noise = [
            GPSMeasurement(
                timestamp=m.timestamp,
                values=m.values,
                covariance=np.diag([100.0**2, 100.0**2]),
                lat=m.lat,
                lon=m.lon,
                accuracy_m=100.0,
            )
            for m in measurements_low_noise
        ]

        kf = VariableNoiseKalmanFilter()
        result_low = kf.filter(measurements_low_noise, cfg)
        # A fresh filter for high-noise (EM state shouldn't carry over)
        kf2 = VariableNoiseKalmanFilter()
        result_high = kf2.filter(high_noise, cfg)

        # Low-noise filter should stay closer to observations
        raw = np.stack([m.values for m in measurements_low_noise])
        dist_low = np.mean(np.linalg.norm(result_low.position_xy() - raw, axis=1))
        dist_high = np.mean(np.linalg.norm(result_high.position_xy() - raw, axis=1))
        # High noise → filter discounts observations → larger deviation from raw
        assert dist_high > dist_low

    def test_empty_measurements_raises(self, simple_df):
        cfg = GPSConfig()
        kf = VariableNoiseKalmanFilter()
        with pytest.raises(ValueError, match="empty"):
            kf.filter([], cfg)

    def test_sensor_type_in_result(self, simple_df):
        measurements, cfg = _make_measurements(simple_df)
        result = VariableNoiseKalmanFilter().filter(measurements, cfg)
        assert result.sensor_type == "gps"
