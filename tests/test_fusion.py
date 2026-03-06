"""Tests for multi-sensor fusion: SensorFuser, SensorFusionPipeline,
and all new sensor types."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kalmangps import (
    CaptureRecaptureConfig,
    CellTowerConfig,
    ConstantVelocity2D,
    CustomSensorConfig,
    FusedPipelineResult,
    GPSConfig,
    SensorFuser,
    SensorFusionPipeline,
)
from kalmangps.sensors.gps import latlon_to_xy
from tests.conftest import make_gps_df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

LAT_REF = 51.5
LON_REF = -0.1


def _gps_stream(n=15, rng_seed=0):
    """Return (measurements, config) for a short GPS trajectory."""
    rng = np.random.default_rng(rng_seed)
    df = make_gps_df(n=n, rng=rng)
    cfg = GPSConfig(lat_ref=None, lon_ref=None)
    meas = cfg.measurements_from_df(df, auto_ref=True)
    return meas, cfg


def _capture_recapture_event(ts: pd.Timestamp, cfg: CaptureRecaptureConfig):
    """Return a single CR measurement using an already-configured config."""
    row = pd.Series({"timestamp": ts})
    return cfg.measurement_from_row(row)


# --------------------------------------------------------------------------- #
# CaptureRecaptureConfig
# --------------------------------------------------------------------------- #

class TestCaptureRecaptureConfig:

    def test_set_reference_required(self):
        cfg = CaptureRecaptureConfig(reader_lat=51.5, reader_lon=-0.1)
        row = pd.Series({"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})
        with pytest.raises(RuntimeError, match="set_reference"):
            cfg.measurement_from_row(row)

    def test_measurement_at_reader_location(self):
        cfg = CaptureRecaptureConfig(reader_lat=51.5, reader_lon=-0.1,
                                      detection_radius_m=20.0)
        cfg.set_reference(51.5, -0.1)
        row = pd.Series({"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})
        m = cfg.measurement_from_row(row)
        # Reader is at the reference point → Cartesian (0, 0)
        assert m.values[0] == pytest.approx(0.0, abs=1e-6)
        assert m.values[1] == pytest.approx(0.0, abs=1e-6)

    def test_covariance_is_radius_squared(self):
        cfg = CaptureRecaptureConfig(reader_lat=51.5, reader_lon=-0.1,
                                      detection_radius_m=30.0)
        cfg.set_reference(51.5, -0.1)
        row = pd.Series({"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})
        m = cfg.measurement_from_row(row)
        assert m.covariance[0, 0] == pytest.approx(900.0)

    def test_confidence_scales_radius(self):
        cfg = CaptureRecaptureConfig(
            reader_lat=51.5, reader_lon=-0.1,
            detection_radius_m=50.0, confidence_col="conf"
        )
        cfg.set_reference(51.5, -0.1)
        high_conf = pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "conf": 1.0
        })
        low_conf = pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"), "conf": 0.5
        })
        m_high = cfg.measurement_from_row(high_conf)
        m_low = cfg.measurement_from_row(low_conf)
        # Lower confidence → larger effective radius → larger covariance
        assert m_low.covariance[0, 0] > m_high.covariance[0, 0]

    def test_observation_matrix_shape(self):
        cfg = CaptureRecaptureConfig(reader_lat=51.5, reader_lon=-0.1)
        assert cfg.observation_matrix.shape == (2, 4)

    def test_sensor_type(self):
        cfg = CaptureRecaptureConfig(reader_lat=51.5, reader_lon=-0.1)
        cfg.set_reference(51.5, -0.1)
        row = pd.Series({"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})
        m = cfg.measurement_from_row(row)
        assert m.sensor_type == "capture_recapture"

    def test_observe_and_jacobian_linear(self):
        cfg = CaptureRecaptureConfig(reader_lat=51.5, reader_lon=-0.1)
        state = np.array([100.0, 200.0, 1.0, 0.5])
        z = cfg.observe(state)
        H = cfg.observation_jacobian(state)
        assert np.allclose(z, H @ state)


# --------------------------------------------------------------------------- #
# CellTowerConfig — range
# --------------------------------------------------------------------------- #

class TestCellTowerConfigRange:

    def _cfg(self, **kw):
        cfg = CellTowerConfig(
            tower_lat=51.5, tower_lon=-0.1,
            measurement_type="range",
            range_std_m=200.0,
            **kw,
        )
        cfg.set_reference(51.5, -0.1)
        return cfg

    def test_set_reference_required(self):
        cfg = CellTowerConfig(tower_lat=51.5, tower_lon=-0.1)
        state = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(RuntimeError):
            cfg.observe(state)

    def test_observe_range_zero_at_tower(self):
        cfg = self._cfg()
        state = np.array([0.0, 0.0, 0.0, 0.0])  # entity at tower
        z = cfg.observe(state)
        # Should return min_range_m (not zero) due to guard
        assert z[0] >= cfg.min_range_m

    def test_observe_range_correct_distance(self):
        cfg = self._cfg()
        # Entity 300 m east of the tower (which is at Cartesian origin)
        state = np.array([300.0, 0.0, 0.0, 0.0])
        z = cfg.observe(state)
        assert z[0] == pytest.approx(300.0, rel=1e-6)

    def test_jacobian_shape(self):
        cfg = self._cfg()
        state = np.array([300.0, 400.0, 0.0, 0.0])
        H = cfg.observation_jacobian(state)
        assert H.shape == (1, 4)

    def test_jacobian_matches_numerical(self):
        cfg = self._cfg()
        state = np.array([300.0, 400.0, 1.0, -0.5])
        H_analytic = cfg.observation_jacobian(state)

        eps = 1e-4
        H_numeric = np.zeros((1, 4))
        for j in range(4):
            sp, sm = state.copy(), state.copy()
            sp[j] += eps; sm[j] -= eps
            H_numeric[0, j] = (cfg.observe(sp)[0] - cfg.observe(sm)[0]) / (2 * eps)

        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)

    def test_rssi_to_range_conversion(self):
        cfg = self._cfg(rssi_at_1m=-40.0, path_loss_exponent=2.0)
        # At 10 m: RSSI = -40 - 20*log10(10) = -60 dBm
        row = pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "rssi_dbm": -60.0,
        })
        m = cfg.measurement_from_row(row)
        assert m.values[0] == pytest.approx(10.0, rel=0.01)

    def test_explicit_range_col_takes_priority(self):
        cfg = self._cfg(range_col="range_m")
        row = pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "rssi_dbm": -90.0,  # would give a large distance
            "range_m": 150.0,
        })
        m = cfg.measurement_from_row(row)
        assert m.values[0] == pytest.approx(150.0)

    def test_observation_matrix_is_none(self):
        cfg = self._cfg()
        assert cfg.observation_matrix is None


# --------------------------------------------------------------------------- #
# CellTowerConfig — angle
# --------------------------------------------------------------------------- #

class TestCellTowerConfigAngle:

    def _cfg(self):
        cfg = CellTowerConfig(
            tower_lat=51.5, tower_lon=-0.1,
            measurement_type="angle",
            angle_std_rad=0.1,
            angle_col="bearing_rad",
        )
        cfg.set_reference(51.5, -0.1)
        return cfg

    def test_observe_angle_correct(self):
        cfg = self._cfg()
        # Entity 100 m north, tower at origin → bearing should be pi/2
        state = np.array([0.0, 100.0, 0.0, 0.0])
        z = cfg.observe(state)
        # atan2(0 - 100, 0 - 0) = atan2(-100, 0) = -pi/2
        expected = math.atan2(cfg._tower_y_m - state[1], cfg._tower_x_m - state[0])
        assert z[0] == pytest.approx(expected, abs=1e-8)

    def test_jacobian_matches_numerical(self):
        cfg = self._cfg()
        state = np.array([200.0, 150.0, 1.0, 0.5])
        H_analytic = cfg.observation_jacobian(state)

        eps = 1e-4
        H_numeric = np.zeros((1, 4))
        for j in range(4):
            sp, sm = state.copy(), state.copy()
            sp[j] += eps; sm[j] -= eps
            H_numeric[0, j] = (cfg.observe(sp)[0] - cfg.observe(sm)[0]) / (2 * eps)

        np.testing.assert_allclose(H_analytic, H_numeric, atol=1e-5)


# --------------------------------------------------------------------------- #
# CellTowerConfig — position
# --------------------------------------------------------------------------- #

class TestCellTowerConfigPosition:

    def test_position_type_is_linear(self):
        cfg = CellTowerConfig(
            tower_lat=51.5, tower_lon=-0.1,
            measurement_type="position",
            position_lat_col="cell_lat",
            position_lon_col="cell_lon",
        )
        cfg.set_reference(51.5, -0.1)
        # observation_jacobian should be constant (linear sensor via override)
        state = np.array([100.0, 200.0, 0.0, 0.0])
        H = cfg.observation_jacobian(state)
        assert H.shape == (2, 4)
        assert H[0, 0] == 1.0
        assert H[1, 1] == 1.0


# --------------------------------------------------------------------------- #
# CustomSensorConfig
# --------------------------------------------------------------------------- #

class TestCustomSensorConfig:

    def test_linear_custom_sensor(self):
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        cfg = CustomSensorConfig(
            name="x_only",
            sensor_type_tag="x_pos",
            observation_matrix_val=H,
            default_covariance=np.array([[25.0]]),
            timestamp_col="ts",
            value_cols=["x_m"],
        )
        state = np.array([42.0, 10.0, 1.0, 0.5])
        assert cfg.observe(state)[0] == pytest.approx(42.0)
        np.testing.assert_array_equal(cfg.observation_jacobian(state), H)

    def test_nonlinear_custom_sensor(self):
        def obs(s): return np.array([np.sqrt(s[0]**2 + s[1]**2)])
        def jac(s):
            d = max(np.sqrt(s[0]**2 + s[1]**2), 1e-6)
            return np.array([[s[0]/d, s[1]/d, 0., 0.]])

        cfg = CustomSensorConfig(
            name="range_from_origin",
            sensor_type_tag="range_origin",
            observe_fn=obs,
            jacobian_fn=jac,
            default_covariance=np.array([[100.0]]),
            timestamp_col="ts",
            value_cols=["range_m"],
        )
        state = np.array([30.0, 40.0, 0.0, 0.0])
        assert cfg.observe(state)[0] == pytest.approx(50.0)

    def test_measurement_from_row(self):
        cfg = CustomSensorConfig(
            name="baro",
            sensor_type_tag="baro",
            observation_matrix_val=np.array([[0., 1., 0., 0.]]),
            default_covariance=np.array([[4.0]]),
            timestamp_col="ts",
            value_cols=["alt_m"],
        )
        row = pd.Series({"ts": pd.Timestamp("2024-01-01", tz="UTC"), "alt_m": 55.0})
        m = cfg.measurement_from_row(row)
        assert m.values[0] == pytest.approx(55.0)
        assert m.covariance[0, 0] == pytest.approx(4.0)

    def test_requires_observe_or_matrix(self):
        with pytest.raises(ValueError, match="observe_fn"):
            CustomSensorConfig(
                name="bad",
                sensor_type_tag="bad",
                default_covariance=np.eye(1),
                value_cols=["x"],
            )


# --------------------------------------------------------------------------- #
# SensorFuser
# --------------------------------------------------------------------------- #

class TestSensorFuser:

    def test_gps_only_fusion(self):
        meas, cfg = _gps_stream(n=15)
        stream = [(m, cfg) for m in meas]
        fuser = SensorFuser()
        result = fuser.run(stream, lat_ref=LAT_REF, lon_ref=LON_REF)
        assert result.n_steps == 15

    def test_result_columns(self):
        meas, cfg = _gps_stream(n=10)
        stream = [(m, cfg) for m in meas]
        result = SensorFuser().run(stream, lat_ref=LAT_REF, lon_ref=LON_REF)
        df = result.to_dataframe()
        for col in ("filtered_lat", "filtered_lon", "filtered_vx_ms",
                    "filtered_vy_ms", "pos_uncertainty_x_m"):
            assert col in df.columns

    def test_empty_stream_returns_empty_result(self):
        result = SensorFuser().run([], lat_ref=LAT_REF, lon_ref=LON_REF)
        assert result.n_steps == 0

    def test_gps_plus_capture_recapture(self):
        rng = np.random.default_rng(42)
        df = make_gps_df(n=20, rng=rng)
        gps_cfg = GPSConfig(lat_ref=None, lon_ref=None)
        gps_meas = gps_cfg.measurements_from_df(df, auto_ref=True)

        lat_ref = gps_cfg.lat_ref
        lon_ref = gps_cfg.lon_ref

        cr_cfg = CaptureRecaptureConfig(
            reader_lat=lat_ref, reader_lon=lon_ref,
            detection_radius_m=25.0,
        )
        cr_cfg.set_reference(lat_ref, lon_ref)

        # One CR event at the midpoint of the trajectory
        mid_ts = df["timestamp"].iloc[10]
        cr_row = pd.Series({"timestamp": mid_ts})
        cr_meas = cr_cfg.measurement_from_row(cr_row)

        stream = [(m, gps_cfg) for m in gps_meas] + [(cr_meas, cr_cfg)]
        result = SensorFuser().run(stream, lat_ref=lat_ref, lon_ref=lon_ref)

        # 20 GPS + 1 CR = 21 steps
        assert result.n_steps == 21
        assert "capture_recapture" in result.sensor_types()

    def test_gps_plus_cell_tower_range(self):
        rng = np.random.default_rng(7)
        df = make_gps_df(n=15, rng=rng)
        gps_cfg = GPSConfig(lat_ref=None, lon_ref=None)
        gps_meas = gps_cfg.measurements_from_df(df, auto_ref=True)

        lat_ref = gps_cfg.lat_ref
        lon_ref = gps_cfg.lon_ref

        tower_cfg = CellTowerConfig(
            tower_lat=lat_ref + 0.01,
            tower_lon=lon_ref + 0.01,
            measurement_type="range",
            range_std_m=300.0,
        )
        tower_cfg.set_reference(lat_ref, lon_ref)

        # Tower measurement at every GPS timestamp
        tower_meas = []
        for m in gps_meas:
            # True range from GPS position to tower
            tx, ty = tower_cfg._tower_x_m, tower_cfg._tower_y_m
            true_range = np.sqrt((m.values[0] - tx)**2 + (m.values[1] - ty)**2)
            from kalmangps.sensors.cell_tower import CellTowerMeasurement
            tm = CellTowerMeasurement(
                timestamp=m.timestamp,
                values=np.array([true_range]),
                covariance=np.array([[300.0**2]]),
                tower_lat=tower_cfg.tower_lat,
                tower_lon=tower_cfg.tower_lon,
                measurement_type="range",
            )
            tower_meas.append(tm)

        stream = (
            [(m, gps_cfg) for m in gps_meas]
            + [(m, tower_cfg) for m in tower_meas]
        )
        result = SensorFuser().run(stream, lat_ref=lat_ref, lon_ref=lon_ref)
        assert result.n_steps == 30
        assert "cell_tower" in result.sensor_types()

    def test_fused_position_close_to_gps(self):
        """Fused GPS+CR result should still be close to raw GPS fixes."""
        rng = np.random.default_rng(1)
        df = make_gps_df(n=20, rng=rng)
        gps_cfg = GPSConfig(lat_ref=None, lon_ref=None)
        gps_meas = gps_cfg.measurements_from_df(df, auto_ref=True)
        lat_ref, lon_ref = gps_cfg.lat_ref, gps_cfg.lon_ref

        cr_cfg = CaptureRecaptureConfig(
            reader_lat=lat_ref, reader_lon=lon_ref,
            detection_radius_m=100.0,
        )
        cr_cfg.set_reference(lat_ref, lon_ref)
        cr_row = pd.Series({"timestamp": df["timestamp"].iloc[5]})
        cr_meas = cr_cfg.measurement_from_row(cr_row)

        stream = [(m, gps_cfg) for m in gps_meas] + [(cr_meas, cr_cfg)]
        result = SensorFuser().run(stream, lat_ref=lat_ref, lon_ref=lon_ref)

        gps_steps = result.at_sensor("gps")
        raw_xy = np.stack([m.values for m in gps_meas])
        fused_xy = np.stack([s.position_xy() for s in gps_steps])
        dists = np.linalg.norm(fused_xy - raw_xy, axis=1)
        assert np.mean(dists) < 100.0  # within 100 m on average

    def test_sensor_types_method(self):
        meas, cfg = _gps_stream(n=8)
        stream = [(m, cfg) for m in meas]
        result = SensorFuser().run(stream, lat_ref=LAT_REF, lon_ref=LON_REF)
        assert result.sensor_types() == {"gps"}

    def test_at_sensor_filter(self):
        meas, cfg = _gps_stream(n=10)
        stream = [(m, cfg) for m in meas]
        result = SensorFuser().run(stream, lat_ref=LAT_REF, lon_ref=LON_REF)
        gps_steps = result.at_sensor("gps")
        assert len(gps_steps) == 10

    def test_unsorted_stream_gets_sorted(self):
        """Fuser should sort the input stream before processing."""
        meas, cfg = _gps_stream(n=10)
        stream = [(m, cfg) for m in reversed(meas)]  # reversed order
        result = SensorFuser().run(stream, lat_ref=LAT_REF, lon_ref=LON_REF)
        timestamps = [s.timestamp for s in result.steps]
        assert timestamps == sorted(timestamps)


# --------------------------------------------------------------------------- #
# SensorFusionPipeline
# --------------------------------------------------------------------------- #

class TestSensorFusionPipeline:

    def _make_pipeline(self, **kw):
        return SensorFusionPipeline(
            sensors={"gps": GPSConfig(lat_col="lat", lon_col="lon",
                                       accuracy_col="accuracy",
                                       timestamp_col="timestamp")},
            primary_sensor="gps",
            **kw,
        )

    def test_gps_only_pipeline(self):
        rng = np.random.default_rng(0)
        df = make_gps_df(n=20, rng=rng)
        result = self._make_pipeline().run(df)
        assert result.n_segments == 1
        assert result.n_steps == 20

    def test_to_dataframe_has_required_columns(self):
        rng = np.random.default_rng(1)
        df = make_gps_df(n=15, rng=rng)
        out = self._make_pipeline().run(df).to_dataframe()
        for col in ("filtered_lat", "filtered_lon", "sensor_type", "timestamp"):
            assert col in out.columns

    def test_multi_trip_splitting(self):
        rng = np.random.default_rng(2)
        df1 = make_gps_df(n=12, trip_id="A", rng=rng)
        df2 = make_gps_df(n=10, trip_id="B", start_lat=48.8, start_lon=2.3, rng=rng)
        df = pd.concat([df1, df2], ignore_index=True)

        pipeline = SensorFusionPipeline(
            sensors={"gps": GPSConfig()},
            primary_sensor="gps",
            group_col="trip_id",
        )
        result = pipeline.run(df)
        assert result.n_segments == 2

    def test_time_gap_splitting(self):
        rng = np.random.default_rng(3)
        early = make_gps_df(n=8, rng=rng, start_time="2024-01-01 08:00:00")
        late = make_gps_df(n=8, rng=rng, start_time="2024-01-01 11:00:00")
        df = pd.concat([early, late], ignore_index=True)

        pipeline = SensorFusionPipeline(
            sensors={"gps": GPSConfig()},
            primary_sensor="gps",
            max_time_gap="5 min",
        )
        result = pipeline.run(df)
        assert result.n_segments == 2

    def test_with_capture_recapture(self):
        rng = np.random.default_rng(4)
        gps_df = make_gps_df(n=15, rng=rng)

        # One CR event in the middle of the trajectory
        mid_ts = gps_df["timestamp"].iloc[7]
        cr_df = pd.DataFrame({
            "timestamp": [mid_ts],
            "reader_id": ["gate_1"],
        })

        # Reader at approximately the same location as the GPS track
        first_lat = float(gps_df["lat"].iloc[0])
        first_lon = float(gps_df["lon"].iloc[0])

        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "gate_1": CaptureRecaptureConfig(
                    reader_lat=first_lat,
                    reader_lon=first_lon,
                    detection_radius_m=50.0,
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"gate_1": cr_df})
        assert result.n_segments == 1
        # 15 GPS + 1 CR = 16 steps
        assert result.n_steps == 16
        sensor_types = result.segment_results[0].fused.sensor_types()
        assert "capture_recapture" in sensor_types

    def test_with_cell_tower(self):
        rng = np.random.default_rng(5)
        gps_df = make_gps_df(n=12, rng=rng)
        lat0 = float(gps_df["lat"].iloc[0])
        lon0 = float(gps_df["lon"].iloc[0])

        # Tower 500 m north
        tower_lat = lat0 + 0.005
        tower_lon = lon0

        tower_df = pd.DataFrame({
            "timestamp": gps_df["timestamp"].values,
            "rssi_dbm": np.full(12, -70.0),
        })

        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "tower_1": CellTowerConfig(
                    tower_lat=tower_lat, tower_lon=tower_lon,
                    measurement_type="range",
                    range_std_m=500.0,
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"tower_1": tower_df})
        assert result.n_segments == 1
        assert result.n_steps == 24  # 12 GPS + 12 tower

    def test_secondary_outside_window_excluded(self):
        """Secondary measurements outside the GPS segment window are dropped."""
        rng = np.random.default_rng(6)
        gps_df = make_gps_df(n=10, start_time="2024-01-01 12:00:00", rng=rng)

        # CR event 3 hours before the GPS segment
        far_ts = pd.Timestamp("2024-01-01 09:00:00", tz="UTC")
        cr_df = pd.DataFrame({"timestamp": [far_ts]})

        first_lat = float(gps_df["lat"].iloc[0])
        first_lon = float(gps_df["lon"].iloc[0])

        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "reader": CaptureRecaptureConfig(
                    reader_lat=first_lat, reader_lon=first_lon,
                    detection_radius_m=30.0,
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"reader": cr_df})
        # Only GPS steps — CR was out of window
        assert result.n_steps == 10

    def test_empty_primary_df(self):
        empty = pd.DataFrame(columns=["timestamp", "lat", "lon", "accuracy"])
        result = self._make_pipeline().run(empty)
        assert result.n_segments == 0

    def test_invalid_primary_sensor_key(self):
        with pytest.raises(ValueError, match="primary_sensor"):
            SensorFusionPipeline(
                sensors={"gps": GPSConfig()},
                primary_sensor="nonexistent",
            )

    def test_fused_pipeline_result_to_dataframe(self):
        rng = np.random.default_rng(9)
        df = make_gps_df(n=12, rng=rng)
        result = self._make_pipeline().run(df)
        out = result.to_dataframe()
        assert len(out) == 12
        assert np.isfinite(out["filtered_lat"]).all()
