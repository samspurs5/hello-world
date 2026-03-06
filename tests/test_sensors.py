"""Tests for GPS sensor config and measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalmangps import GPSConfig, GPSMeasurement
from kalmangps.sensors.gps import latlon_to_xy, xy_to_latlon


class TestGPSConfig:

    def _row(self, lat=51.5, lon=-0.1, accuracy=5.0, ts="2024-01-01 12:00:00"):
        return pd.Series({
            "lat": lat,
            "lon": lon,
            "accuracy": accuracy,
            "timestamp": pd.Timestamp(ts, tz="UTC"),
        })

    def test_measurement_from_row_values_shape(self):
        cfg = GPSConfig(lat_ref=51.5, lon_ref=-0.1)
        m = cfg.measurement_from_row(self._row())
        assert m.values.shape == (2,)

    def test_measurement_from_row_covariance_shape(self):
        cfg = GPSConfig(lat_ref=51.5, lon_ref=-0.1)
        m = cfg.measurement_from_row(self._row(accuracy=8.0))
        assert m.covariance.shape == (2, 2)
        assert m.covariance[0, 0] == pytest.approx(64.0)

    def test_measurement_uses_default_accuracy_when_nan(self):
        cfg = GPSConfig(lat_ref=51.5, lon_ref=-0.1, default_accuracy_m=20.0)
        row = self._row()
        row["accuracy"] = float("nan")
        m = cfg.measurement_from_row(row)
        assert m.accuracy_m == pytest.approx(20.0)

    def test_measurement_uses_default_when_col_missing(self):
        cfg = GPSConfig(lat_ref=51.5, lon_ref=-0.1, default_accuracy_m=15.0)
        row = pd.Series({
            "lat": 51.5, "lon": -0.1,
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        })
        m = cfg.measurement_from_row(row)
        assert m.accuracy_m == pytest.approx(15.0)

    def test_auto_ref_set_from_first_row(self):
        import pandas as pd
        from tests.conftest import make_gps_df
        df = make_gps_df(n=5)
        cfg = GPSConfig(lat_ref=None, lon_ref=None)
        measurements = cfg.measurements_from_df(df, auto_ref=True)
        # Reference should now be set
        assert cfg.lat_ref is not None
        assert cfg.lon_ref is not None
        # First measurement should be at (0, 0) in local coords
        assert measurements[0].values[0] == pytest.approx(0.0, abs=1e-9)
        assert measurements[0].values[1] == pytest.approx(0.0, abs=1e-9)

    def test_observation_matrix_shape(self):
        cfg = GPSConfig()
        H = cfg.observation_matrix
        assert H.shape == (2, 4)
        # Maps x and y directly (first two state dims)
        assert H[0, 0] == 1.0
        assert H[1, 1] == 1.0
        assert H[0, 2] == 0.0
        assert H[1, 3] == 0.0

    def test_gps_measurement_sensor_type(self):
        cfg = GPSConfig(lat_ref=51.0, lon_ref=0.0)
        row = self._row()
        m = cfg.measurement_from_row(row)
        assert m.sensor_type == "gps"

    def test_n_obs(self):
        cfg = GPSConfig(lat_ref=51.0, lon_ref=0.0)
        m = cfg.measurement_from_row(self._row())
        assert m.n_obs == 2
