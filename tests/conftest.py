"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_gps_df(
    n: int = 20,
    start_lat: float = 51.5,
    start_lon: float = -0.1,
    trip_id: str = "trip_1",
    start_time: str = "2024-01-01 12:00:00",
    dt_seconds: float = 5.0,
    noise_std: float = 0.00005,
    accuracy: float = 5.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate a synthetic GPS trajectory for testing."""
    if rng is None:
        rng = np.random.default_rng(42)

    times = pd.date_range(start_time, periods=n, freq=f"{int(dt_seconds)}s", tz="UTC")
    lats = start_lat + np.cumsum(rng.normal(0, noise_std, n))
    lons = start_lon + np.cumsum(rng.normal(0, noise_std, n))
    accuracies = rng.uniform(accuracy * 0.5, accuracy * 2.0, n)

    return pd.DataFrame({
        "timestamp": times,
        "lat": lats,
        "lon": lons,
        "accuracy": accuracies,
        "trip_id": trip_id,
    })


@pytest.fixture
def simple_df():
    """20-point single-trip trajectory."""
    return make_gps_df(n=20)


@pytest.fixture
def multi_trip_df():
    """Two trips concatenated."""
    rng = np.random.default_rng(0)
    df1 = make_gps_df(n=15, trip_id="A", rng=rng)
    df2 = make_gps_df(n=10, trip_id="B", start_lat=48.8, start_lon=2.3, rng=rng)
    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def gapped_df():
    """Single trip with a large time gap in the middle."""
    rng = np.random.default_rng(7)
    first = make_gps_df(n=10, rng=rng, start_time="2024-01-01 08:00:00")
    # 2-hour gap
    second = make_gps_df(n=10, rng=rng, start_time="2024-01-01 10:00:00")
    return pd.concat([first, second], ignore_index=True)
