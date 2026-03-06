"""Integration tests for GPSKalmanPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kalmangps import GPSConfig, GPSKalmanPipeline
from kalmangps.state import ConstantVelocity2D
from tests.conftest import make_gps_df


class TestGPSKalmanPipeline:

    def test_run_returns_pipeline_result(self, simple_df):
        pipeline = GPSKalmanPipeline()
        result = pipeline.run(simple_df)
        assert result.n_segments >= 1
        assert result.n_points > 0

    def test_to_dataframe_has_expected_columns(self, simple_df):
        pipeline = GPSKalmanPipeline()
        df_out = pipeline.run(simple_df).to_dataframe()
        for col in ("filtered_lat", "filtered_lon", "filtered_vx_ms",
                    "filtered_vy_ms", "pos_uncertainty_x_m", "pos_uncertainty_y_m"):
            assert col in df_out.columns, f"Missing column: {col}"

    def test_output_row_count_matches_input(self, simple_df):
        pipeline = GPSKalmanPipeline(min_segment_length=1)
        result = pipeline.run(simple_df)
        assert result.n_points == len(simple_df)

    def test_multi_trip_produces_multiple_segments(self, multi_trip_df):
        pipeline = GPSKalmanPipeline(group_col="trip_id")
        result = pipeline.run(multi_trip_df)
        assert result.n_segments == 2

    def test_time_gap_split_through_pipeline(self, gapped_df):
        pipeline = GPSKalmanPipeline(max_time_gap="5 min")
        result = pipeline.run(gapped_df)
        assert result.n_segments == 2

    def test_filtered_coords_are_finite(self, simple_df):
        df_out = GPSKalmanPipeline().run(simple_df).to_dataframe()
        assert np.isfinite(df_out["filtered_lat"]).all()
        assert np.isfinite(df_out["filtered_lon"]).all()

    def test_filtered_lat_close_to_raw(self, simple_df):
        """Filtered lat should be within 0.01° of the raw lat (≈1 km)."""
        df_out = GPSKalmanPipeline().run(simple_df).to_dataframe()
        delta = (df_out["filtered_lat"] - df_out["lat"]).abs()
        assert delta.max() < 0.01

    def test_filtered_lon_close_to_raw(self, simple_df):
        df_out = GPSKalmanPipeline().run(simple_df).to_dataframe()
        delta = (df_out["filtered_lon"] - df_out["lon"]).abs()
        assert delta.max() < 0.01

    def test_custom_gps_config_column_names(self):
        rng = np.random.default_rng(5)
        df = make_gps_df(n=12, rng=rng).rename(columns={
            "lat": "latitude",
            "lon": "longitude",
            "accuracy": "h_acc",
            "timestamp": "utc",
        })
        cfg = GPSConfig(lat_col="latitude", lon_col="longitude",
                        accuracy_col="h_acc", timestamp_col="utc")
        pipeline = GPSKalmanPipeline(gps_config=cfg)
        result = pipeline.run(df)
        assert result.n_segments == 1

    def test_empty_dataframe_returns_empty_result(self):
        import pandas as pd
        empty = pd.DataFrame(columns=["timestamp", "lat", "lon", "accuracy"])
        pipeline = GPSKalmanPipeline()
        result = pipeline.run(empty)
        assert result.n_segments == 0

    def test_custom_process_noise_affects_smoothing(self, simple_df):
        """Higher process noise → filter tracks observations more closely."""
        low_model = ConstantVelocity2D(process_noise_std=0.01)
        high_model = ConstantVelocity2D(process_noise_std=100.0)

        df_low = GPSKalmanPipeline(state_model=low_model).run(simple_df).to_dataframe()
        df_high = GPSKalmanPipeline(state_model=high_model).run(simple_df).to_dataframe()

        # High process noise → less smoothing → closer to raw lat
        err_low = (df_low["filtered_lat"] - simple_df["lat"]).abs().mean()
        err_high = (df_high["filtered_lat"] - simple_df["lat"]).abs().mean()
        assert err_high < err_low

    def test_no_accuracy_column_uses_default(self):
        rng = np.random.default_rng(3)
        df = make_gps_df(n=12, rng=rng).drop(columns=["accuracy"])
        cfg = GPSConfig(accuracy_col=None)
        pipeline = GPSKalmanPipeline(gps_config=cfg)
        result = pipeline.run(df)
        assert result.n_segments == 1
        df_out = result.to_dataframe()
        assert np.isfinite(df_out["filtered_lat"]).all()

    def test_group_key_in_output(self, multi_trip_df):
        df_out = GPSKalmanPipeline(group_col="trip_id").run(multi_trip_df).to_dataframe()
        assert set(df_out["group_key"]) == {"A", "B"}
