"""Tests for multi-sensor fusion: models, SensorFactory, SensorFuser,
and SensorFusionPipeline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from kalmangps import (
    BearingModel,
    CallableModel,
    ConstantVelocity2D,
    FixedPointModel,
    FusedPipelineResult,
    GPSConfig,
    GenericSensorConfig,
    RangeModel,
    SensorFactory,
    SensorFuser,
    SensorFusionPipeline,
)
from kalmangps.sensors.gps import latlon_to_xy
from tests.conftest import make_gps_df

LAT_REF = 51.5
LON_REF = -0.1


def _gps_stream(n=15, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    df = make_gps_df(n=n, rng=rng)
    cfg = GPSConfig(lat_ref=None, lon_ref=None)
    meas = cfg.measurements_from_df(df, auto_ref=True)
    return meas, cfg


# --------------------------------------------------------------------------- #
# ObservationModel — FixedPointModel
# --------------------------------------------------------------------------- #

class TestFixedPointModel:

    def _model(self, **kw):
        m = FixedPointModel(lat=LAT_REF, lon=LON_REF, **kw)
        m.set_reference(LAT_REF, LON_REF)
        return m

    def test_requires_set_reference(self):
        m = FixedPointModel(lat=LAT_REF, lon=LON_REF)
        with pytest.raises(RuntimeError, match="set_reference"):
            m.default_values()

    def test_default_values_at_origin(self):
        m = self._model()
        v = m.default_values()
        assert v[0] == pytest.approx(0.0, abs=1e-6)
        assert v[1] == pytest.approx(0.0, abs=1e-6)

    def test_observe_returns_state_xy(self):
        m = self._model()
        state = np.array([42.0, 17.0, 1.0, 0.5])
        z = m.observe(state)
        # h(x) = H @ x = [x, y]
        assert z[0] == pytest.approx(42.0)
        assert z[1] == pytest.approx(17.0)

    def test_jacobian_is_constant(self):
        m = self._model()
        H1 = m.jacobian(np.zeros(4))
        H2 = m.jacobian(np.array([100.0, 200.0, 1.0, 0.0]))
        np.testing.assert_array_equal(H1, H2)
        assert H1.shape == (2, 4)

    def test_covariance_radius_squared(self):
        m = self._model(uncertainty_m=30.0)
        cov = m.default_covariance(pd.Series({}))
        assert cov[0, 0] == pytest.approx(900.0)

    def test_confidence_scales_covariance(self):
        m = FixedPointModel(lat=LAT_REF, lon=LON_REF,
                            uncertainty_m=50.0, confidence_col="conf")
        m.set_reference(LAT_REF, LON_REF)
        high = m.default_covariance(pd.Series({"conf": 1.0}))
        low  = m.default_covariance(pd.Series({"conf": 0.5}))
        assert low[0, 0] > high[0, 0]

    def test_n_obs(self):
        assert self._model().n_obs == 2


# --------------------------------------------------------------------------- #
# ObservationModel — RangeModel
# --------------------------------------------------------------------------- #

class TestRangeModel:

    def _model(self, **kw):
        m = RangeModel(lat=LAT_REF, lon=LON_REF, **kw)
        m.set_reference(LAT_REF, LON_REF)
        return m

    def test_requires_set_reference(self):
        m = RangeModel(lat=LAT_REF, lon=LON_REF)
        with pytest.raises(RuntimeError):
            m.observe(np.zeros(4))

    def test_range_correct_distance(self):
        m = self._model()
        state = np.array([300.0, 0.0, 0.0, 0.0])
        assert m.observe(state)[0] == pytest.approx(300.0, rel=1e-6)

    def test_range_clamps_at_min(self):
        m = self._model(min_range_m=10.0)
        state = np.zeros(4)  # entity at node
        assert m.observe(state)[0] >= 10.0

    def test_jacobian_shape(self):
        m = self._model()
        assert m.jacobian(np.array([300.0, 400.0, 0.0, 0.0])).shape == (1, 4)

    def test_jacobian_matches_numerical(self):
        m = self._model()
        state = np.array([300.0, 400.0, 1.0, -0.5])
        H_a = m.jacobian(state)
        eps = 1e-4
        H_n = np.zeros((1, 4))
        for j in range(4):
            sp, sm = state.copy(), state.copy()
            sp[j] += eps; sm[j] -= eps
            H_n[0, j] = (m.observe(sp)[0] - m.observe(sm)[0]) / (2 * eps)
        np.testing.assert_allclose(H_a, H_n, atol=1e-5)

    def test_rssi_to_range(self):
        m = self._model(rssi_at_1m=-40.0, path_loss_exponent=2.0)
        # RSSI = -40 - 20*log10(10) = -60 dBm → 10 m
        assert m.rssi_to_range(-60.0) == pytest.approx(10.0, rel=0.01)

    def test_n_obs(self):
        assert self._model().n_obs == 1


# --------------------------------------------------------------------------- #
# ObservationModel — BearingModel
# --------------------------------------------------------------------------- #

class TestBearingModel:

    def _model(self, **kw):
        m = BearingModel(lat=LAT_REF, lon=LON_REF, **kw)
        m.set_reference(LAT_REF, LON_REF)
        return m

    def test_bearing_correct(self):
        m = self._model()
        # entity 100 m north of node → bearing = atan2(0-100, 0-0) = -pi/2
        state = np.array([0.0, 100.0, 0.0, 0.0])
        expected = math.atan2(m._y_m - state[1], m._x_m - state[0])
        assert m.observe(state)[0] == pytest.approx(expected, abs=1e-8)

    def test_jacobian_matches_numerical(self):
        m = self._model()
        state = np.array([200.0, 150.0, 1.0, 0.5])
        H_a = m.jacobian(state)
        eps = 1e-4
        H_n = np.zeros((1, 4))
        for j in range(4):
            sp, sm = state.copy(), state.copy()
            sp[j] += eps; sm[j] -= eps
            H_n[0, j] = (m.observe(sp)[0] - m.observe(sm)[0]) / (2 * eps)
        np.testing.assert_allclose(H_a, H_n, atol=1e-5)

    def test_n_obs(self):
        assert self._model().n_obs == 1


# --------------------------------------------------------------------------- #
# ObservationModel — CallableModel
# --------------------------------------------------------------------------- #

class TestCallableModel:

    def test_linear_via_matrix(self):
        H = np.array([[1.0, 0.0, 0.0, 0.0]])
        m = CallableModel(observation_matrix_val=H, default_cov=np.array([[25.0]]))
        state = np.array([42.0, 10.0, 1.0, 0.5])
        assert m.observe(state)[0] == pytest.approx(42.0)
        np.testing.assert_array_equal(m.jacobian(state), H)

    def test_nonlinear_callable(self):
        def obs(s): return np.array([np.sqrt(s[0]**2 + s[1]**2)])
        def jac(s):
            d = max(np.sqrt(s[0]**2 + s[1]**2), 1e-6)
            return np.array([[s[0]/d, s[1]/d, 0., 0.]])
        m = CallableModel(observe_fn=obs, jacobian_fn=jac,
                          _n_obs=1, default_cov=np.array([[100.0]]))
        state = np.array([30.0, 40.0, 0.0, 0.0])
        assert m.observe(state)[0] == pytest.approx(50.0)

    def test_requires_either_observe_or_matrix(self):
        with pytest.raises(ValueError):
            CallableModel(default_cov=np.eye(1))


# --------------------------------------------------------------------------- #
# SensorFactory
# --------------------------------------------------------------------------- #

class TestSensorFactory:

    def test_fixed_point_creates_config(self):
        cfg = SensorFactory.fixed_point(lat=51.5, lon=-0.1, uncertainty_m=30.0,
                                         name="cam_1")
        assert isinstance(cfg, GenericSensorConfig)
        assert cfg.name == "cam_1"
        assert isinstance(cfg.model, FixedPointModel)

    def test_range_sensor_creates_config(self):
        cfg = SensorFactory.range_sensor(lat=51.5, lon=-0.1, range_std_m=200.0,
                                          name="tower_1")
        assert isinstance(cfg, GenericSensorConfig)
        assert isinstance(cfg.model, RangeModel)

    def test_bearing_sensor_creates_config(self):
        cfg = SensorFactory.bearing_sensor(lat=51.5, lon=-0.1, angle_std_rad=0.1,
                                            name="aoa_1")
        assert isinstance(cfg, GenericSensorConfig)
        assert isinstance(cfg.model, BearingModel)

    def test_callable_sensor_creates_config(self):
        H = np.array([[0., 1., 0., 0.]])
        cfg = SensorFactory.callable_sensor(
            observe_fn=None,
            observation_matrix=H,
            default_covariance=np.array([[4.0]]),
            value_extractor=lambda row: np.array([row["alt"]]),
            name="baro",
        )
        assert isinstance(cfg, GenericSensorConfig)
        assert isinstance(cfg.model, CallableModel)

    def test_sensor_tag_defaults_to_name(self):
        cfg = SensorFactory.fixed_point(lat=51.5, lon=-0.1, name="gate_A")
        assert cfg.sensor_tag == "gate_A"

    def test_sensor_tag_override(self):
        cfg = SensorFactory.fixed_point(lat=51.5, lon=-0.1,
                                         name="gate_A", sensor_tag="rfid")
        assert cfg.sensor_tag == "rfid"

    def test_register_and_create(self):
        @SensorFactory.register("test_ble")
        def _ble(lat, lon, **kwargs):
            return SensorFactory.range_sensor(
                lat, lon, rssi_at_1m=-59, path_loss_exponent=2.0, **kwargs
            )

        cfg = SensorFactory.create("test_ble", lat=51.5, lon=-0.1,
                                    name="beacon_1")
        assert isinstance(cfg.model, RangeModel)
        assert cfg.model.rssi_at_1m == pytest.approx(-59.0)

    def test_create_unknown_type_raises(self):
        with pytest.raises(KeyError, match="Unknown sensor type"):
            SensorFactory.create("does_not_exist")

    def test_registered_types_includes_builtins(self):
        types = SensorFactory.registered_types()
        assert "fixed_point" in types
        assert "range" in types
        assert "bearing" in types
        assert "callable" in types

    def test_fixed_point_measurement_from_row(self):
        cfg = SensorFactory.fixed_point(lat=LAT_REF, lon=LON_REF,
                                         uncertainty_m=20.0, name="cam")
        cfg.set_reference(LAT_REF, LON_REF)
        row = pd.Series({"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})
        m = cfg.measurement_from_row(row)
        # sensor at reference → Cartesian (0, 0)
        assert m.values[0] == pytest.approx(0.0, abs=1e-6)
        assert m.values[1] == pytest.approx(0.0, abs=1e-6)
        assert m.covariance[0, 0] == pytest.approx(400.0)

    def test_range_sensor_measurement_from_rssi(self):
        cfg = SensorFactory.range_sensor(lat=LAT_REF, lon=LON_REF,
                                          rssi_at_1m=-40.0,
                                          path_loss_exponent=2.0,
                                          name="tower")
        cfg.set_reference(LAT_REF, LON_REF)
        row = pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "rssi_dbm": -60.0,   # → 10 m
        })
        m = cfg.measurement_from_row(row)
        assert m.values[0] == pytest.approx(10.0, rel=0.01)

    def test_sensor_type_tag_in_measurement(self):
        cfg = SensorFactory.fixed_point(lat=LAT_REF, lon=LON_REF, name="entrance_cam")
        cfg.set_reference(LAT_REF, LON_REF)
        row = pd.Series({"timestamp": pd.Timestamp("2024-01-01", tz="UTC")})
        m = cfg.measurement_from_row(row)
        assert m.sensor_type == "entrance_cam"

    def test_bearing_sensor_measurement_from_row(self):
        cfg = SensorFactory.bearing_sensor(lat=LAT_REF, lon=LON_REF,
                                            angle_col="bearing",
                                            name="aoa")
        cfg.set_reference(LAT_REF, LON_REF)
        row = pd.Series({
            "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
            "bearing": 1.57,
        })
        m = cfg.measurement_from_row(row)
        assert m.values[0] == pytest.approx(1.57)


# --------------------------------------------------------------------------- #
# SensorFuser
# --------------------------------------------------------------------------- #

class TestSensorFuser:

    def test_gps_only_fusion(self):
        meas, cfg = _gps_stream(n=15)
        result = SensorFuser().run([(m, cfg) for m in meas],
                                   lat_ref=LAT_REF, lon_ref=LON_REF)
        assert result.n_steps == 15

    def test_result_to_dataframe_columns(self):
        meas, cfg = _gps_stream(n=10)
        result = SensorFuser().run([(m, cfg) for m in meas],
                                   lat_ref=LAT_REF, lon_ref=LON_REF)
        df = result.to_dataframe()
        for col in ("filtered_lat", "filtered_lon", "filtered_vx_ms",
                    "filtered_vy_ms", "sensor_type"):
            assert col in df.columns

    def test_empty_stream(self):
        result = SensorFuser().run([], lat_ref=LAT_REF, lon_ref=LON_REF)
        assert result.n_steps == 0

    def test_gps_plus_fixed_point(self):
        rng = np.random.default_rng(42)
        df = make_gps_df(n=20, rng=rng)
        gps_cfg = GPSConfig(lat_ref=None, lon_ref=None)
        gps_meas = gps_cfg.measurements_from_df(df, auto_ref=True)
        lat_ref, lon_ref = gps_cfg.lat_ref, gps_cfg.lon_ref

        cam = SensorFactory.fixed_point(lat=lat_ref, lon=lon_ref,
                                         uncertainty_m=25.0, name="cam_1")
        cam.set_reference(lat_ref, lon_ref)
        cr_row = pd.Series({"timestamp": df["timestamp"].iloc[10]})
        cr_meas = cam.measurement_from_row(cr_row)

        stream = [(m, gps_cfg) for m in gps_meas] + [(cr_meas, cam)]
        result = SensorFuser().run(stream, lat_ref=lat_ref, lon_ref=lon_ref)
        assert result.n_steps == 21
        assert "cam_1" in result.sensor_types()

    def test_gps_plus_range_sensor(self):
        rng = np.random.default_rng(7)
        df = make_gps_df(n=15, rng=rng)
        gps_cfg = GPSConfig(lat_ref=None, lon_ref=None)
        gps_meas = gps_cfg.measurements_from_df(df, auto_ref=True)
        lat_ref, lon_ref = gps_cfg.lat_ref, gps_cfg.lon_ref

        tower = SensorFactory.range_sensor(
            lat=lat_ref + 0.01, lon=lon_ref + 0.01,
            range_std_m=300.0, name="tower_1"
        )
        tower.set_reference(lat_ref, lon_ref)

        from kalmangps.sensors.generic import GenericMeasurement, _measurement_class
        tower_meas = []
        for m in gps_meas:
            tx = tower.model._x_m
            ty = tower.model._y_m
            true_range = np.sqrt((m.values[0] - tx)**2 + (m.values[1] - ty)**2)
            cls = _measurement_class("tower_1")
            tm = cls(
                timestamp=m.timestamp,
                values=np.array([true_range]),
                covariance=np.array([[300.0**2]]),
            )
            tower_meas.append(tm)

        stream = ([(m, gps_cfg) for m in gps_meas]
                  + [(m, tower) for m in tower_meas])
        result = SensorFuser().run(stream, lat_ref=lat_ref, lon_ref=lon_ref)
        assert result.n_steps == 30
        assert "tower_1" in result.sensor_types()

    def test_fused_position_close_to_gps(self):
        rng = np.random.default_rng(1)
        df = make_gps_df(n=20, rng=rng)
        gps_cfg = GPSConfig(lat_ref=None, lon_ref=None)
        gps_meas = gps_cfg.measurements_from_df(df, auto_ref=True)
        lat_ref, lon_ref = gps_cfg.lat_ref, gps_cfg.lon_ref

        cam = SensorFactory.fixed_point(lat=lat_ref, lon=lon_ref,
                                         uncertainty_m=100.0, name="cam")
        cam.set_reference(lat_ref, lon_ref)
        cr_meas = cam.measurement_from_row(
            pd.Series({"timestamp": df["timestamp"].iloc[5]})
        )

        stream = [(m, gps_cfg) for m in gps_meas] + [(cr_meas, cam)]
        result = SensorFuser().run(stream, lat_ref=lat_ref, lon_ref=lon_ref)
        gps_steps = result.at_sensor("gps")
        raw_xy = np.stack([m.values for m in gps_meas])
        fused_xy = np.stack([s.position_xy() for s in gps_steps])
        dists = np.linalg.norm(fused_xy - raw_xy, axis=1)
        assert np.mean(dists) < 100.0

    def test_unsorted_stream_sorted_internally(self):
        meas, cfg = _gps_stream(n=10)
        stream = list(reversed([(m, cfg) for m in meas]))
        result = SensorFuser().run(stream, lat_ref=LAT_REF, lon_ref=LON_REF)
        ts = [s.timestamp for s in result.steps]
        assert ts == sorted(ts)

    def test_at_sensor_filter(self):
        meas, cfg = _gps_stream(n=10)
        result = SensorFuser().run([(m, cfg) for m in meas],
                                   lat_ref=LAT_REF, lon_ref=LON_REF)
        assert len(result.at_sensor("gps")) == 10


# --------------------------------------------------------------------------- #
# SensorFusionPipeline
# --------------------------------------------------------------------------- #

class TestSensorFusionPipeline:

    def _pipeline(self, **kw):
        return SensorFusionPipeline(
            sensors={"gps": GPSConfig()},
            primary_sensor="gps",
            **kw,
        )

    def test_gps_only(self):
        df = make_gps_df(n=20)
        result = self._pipeline().run(df)
        assert result.n_segments == 1
        assert result.n_steps == 20

    def test_to_dataframe_columns(self):
        df = make_gps_df(n=15)
        out = self._pipeline().run(df).to_dataframe()
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
        assert pipeline.run(df).n_segments == 2

    def test_time_gap_splitting(self):
        rng = np.random.default_rng(3)
        early = make_gps_df(n=8, rng=rng, start_time="2024-01-01 08:00:00")
        late  = make_gps_df(n=8, rng=rng, start_time="2024-01-01 11:00:00")
        df = pd.concat([early, late], ignore_index=True)
        pipeline = SensorFusionPipeline(
            sensors={"gps": GPSConfig()},
            primary_sensor="gps",
            max_time_gap="5 min",
        )
        assert pipeline.run(df).n_segments == 2

    def test_with_fixed_point_camera(self):
        rng = np.random.default_rng(4)
        gps_df = make_gps_df(n=15, rng=rng)
        mid_ts = gps_df["timestamp"].iloc[7]
        cam_df = pd.DataFrame({"timestamp": [mid_ts]})

        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "entrance_cam": SensorFactory.fixed_point(
                    lat=float(gps_df["lat"].iloc[0]),
                    lon=float(gps_df["lon"].iloc[0]),
                    uncertainty_m=50.0,
                    name="entrance_cam",
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"entrance_cam": cam_df})
        assert result.n_segments == 1
        assert result.n_steps == 16  # 15 GPS + 1 camera
        assert "entrance_cam" in result.segment_results[0].fused.sensor_types()

    def test_with_range_sensor(self):
        rng = np.random.default_rng(5)
        gps_df = make_gps_df(n=12, rng=rng)
        lat0 = float(gps_df["lat"].iloc[0])
        lon0 = float(gps_df["lon"].iloc[0])
        tower_df = pd.DataFrame({
            "timestamp": gps_df["timestamp"].values,
            "rssi_dbm": np.full(12, -70.0),
        })
        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "tower_1": SensorFactory.range_sensor(
                    lat=lat0 + 0.005, lon=lon0, range_std_m=500.0, name="tower_1"
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"tower_1": tower_df})
        assert result.n_steps == 24

    def test_secondary_outside_window_excluded(self):
        rng = np.random.default_rng(6)
        gps_df = make_gps_df(n=10, start_time="2024-01-01 12:00:00", rng=rng)
        far_df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-01 09:00:00", tz="UTC")]
        })
        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "gate": SensorFactory.fixed_point(
                    lat=float(gps_df["lat"].iloc[0]),
                    lon=float(gps_df["lon"].iloc[0]),
                    uncertainty_m=30.0, name="gate",
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"gate": far_df})
        assert result.n_steps == 10  # only GPS

    def test_callable_sensor_via_factory(self):
        """Verify that a callable_sensor integrates end-to-end."""
        rng = np.random.default_rng(8)
        gps_df = make_gps_df(n=10, rng=rng)
        # Fake a "north position" sensor that measures y directly
        H = np.array([[0., 1., 0., 0.]])
        north_df = pd.DataFrame({
            "timestamp": gps_df["timestamp"].values,
            "north_m": np.random.default_rng(9).normal(0, 5, 10),
        })
        pipeline = SensorFusionPipeline(
            sensors={
                "gps": GPSConfig(),
                "north_sensor": SensorFactory.callable_sensor(
                    observe_fn=None,
                    observation_matrix=H,
                    default_covariance=np.array([[25.0]]),
                    value_extractor=lambda row: np.array([float(row["north_m"])]),
                    name="north_sensor",
                ),
            },
            primary_sensor="gps",
        )
        result = pipeline.run(gps_df, secondary_dfs={"north_sensor": north_df})
        assert result.n_steps == 20

    def test_empty_primary_df(self):
        empty = pd.DataFrame(columns=["timestamp", "lat", "lon", "accuracy"])
        result = self._pipeline().run(empty)
        assert result.n_segments == 0

    def test_invalid_primary_sensor_raises(self):
        with pytest.raises(ValueError, match="primary_sensor"):
            SensorFusionPipeline(
                sensors={"gps": GPSConfig()},
                primary_sensor="nonexistent",
            )

    def test_result_to_dataframe_finite(self):
        df = make_gps_df(n=12)
        out = self._pipeline().run(df).to_dataframe()
        assert np.isfinite(out["filtered_lat"]).all()
        assert np.isfinite(out["filtered_lon"]).all()
