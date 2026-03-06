"""Tests for TrajectorySplitter."""

from __future__ import annotations

import pandas as pd
import pytest

from kalmangps import TrajectorySplitter, TrajectorySegment
from tests.conftest import make_gps_df


class TestTrajectorySplitter:

    def test_no_group_col_returns_single_group(self, simple_df):
        splitter = TrajectorySplitter(max_time_gap="10 min")
        segs = splitter.split(simple_df)
        assert len(segs) == 1
        assert segs[0].group_key is None
        assert segs[0].n_points == len(simple_df)

    def test_group_col_produces_one_segment_per_trip(self, multi_trip_df):
        splitter = TrajectorySplitter(group_col="trip_id", max_time_gap="10 min")
        segs = splitter.split(multi_trip_df)
        assert len(segs) == 2
        keys = {s.group_key for s in segs}
        assert keys == {"A", "B"}

    def test_time_gap_splits_single_trip(self, gapped_df):
        # gap is 2 hours, threshold is 5 min → should split into 2 segments
        splitter = TrajectorySplitter(max_time_gap="5 min")
        segs = splitter.split(gapped_df)
        assert len(segs) == 2

    def test_time_gap_no_split_when_gap_is_small(self, simple_df):
        # All fixes are 5s apart; threshold 10 min → no split
        splitter = TrajectorySplitter(max_time_gap="10 min")
        segs = splitter.split(simple_df)
        assert len(segs) == 1

    def test_min_segment_length_drops_short_segments(self):
        rng_df = make_gps_df(n=2)  # only 2 points
        splitter = TrajectorySplitter(min_segment_length=3)
        segs = splitter.split(rng_df)
        assert len(segs) == 0

    def test_segment_data_is_sorted_by_timestamp(self, simple_df):
        shuffled = simple_df.sample(frac=1, random_state=42).reset_index(drop=True)
        splitter = TrajectorySplitter(max_time_gap="1 h")
        segs = splitter.split(shuffled)
        assert len(segs) == 1
        times = segs[0].data["timestamp"].tolist()
        assert times == sorted(times)

    def test_missing_timestamp_col_raises(self, simple_df):
        splitter = TrajectorySplitter(timestamp_col="nonexistent")
        with pytest.raises(ValueError, match="timestamp column"):
            splitter.split(simple_df)

    def test_missing_group_col_raises(self, simple_df):
        splitter = TrajectorySplitter(group_col="nonexistent")
        with pytest.raises(ValueError, match="group column"):
            splitter.split(simple_df)

    def test_group_and_time_split_combined(self):
        """Each trip should be split on its own time gap independently."""
        import numpy as np
        rng = np.random.default_rng(1)
        early = make_gps_df(n=8, trip_id="X", start_time="2024-01-01 08:00:00", rng=rng)
        late = make_gps_df(n=8, trip_id="X", start_time="2024-01-01 11:00:00", rng=rng)
        other = make_gps_df(n=8, trip_id="Y", start_time="2024-01-01 09:00:00", rng=rng)
        df = pd.concat([early, late, other], ignore_index=True)

        splitter = TrajectorySplitter(group_col="trip_id", max_time_gap="5 min")
        segs = splitter.split(df)
        # trip X splits into 2, trip Y stays as 1 → total 3
        assert len(segs) == 3

    def test_segment_index_is_zero_based_per_group(self):
        import numpy as np
        rng = np.random.default_rng(2)
        early = make_gps_df(n=6, trip_id="Z", start_time="2024-01-01 08:00:00", rng=rng)
        late = make_gps_df(n=6, trip_id="Z", start_time="2024-01-01 11:00:00", rng=rng)
        df = pd.concat([early, late], ignore_index=True)

        splitter = TrajectorySplitter(group_col="trip_id", max_time_gap="5 min")
        segs = splitter.split(df)
        assert len(segs) == 2
        indices = sorted(s.segment_index for s in segs)
        assert indices == [0, 1]
