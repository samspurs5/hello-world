"""kalmangps — Kalman filter pipeline for GPS trajectories.

Quick start
-----------
>>> import pandas as pd
>>> from kalmangps import GPSKalmanPipeline, GPSConfig
>>>
>>> df = pd.read_csv("gps_log.csv")
>>> pipeline = GPSKalmanPipeline(
...     gps_config=GPSConfig(
...         lat_col="lat",
...         lon_col="lon",
...         accuracy_col="h_accuracy",   # per-fix accuracy in metres
...         timestamp_col="utc_time",
...     ),
...     group_col="trip_id",             # split by this column first
...     max_time_gap="5 min",            # then split on time gaps > 5 min
... )
>>> result = pipeline.run(df)
>>> smoothed = result.to_dataframe()    # filtered_lat, filtered_lon, …
"""

from .filters import KalmanFilterResult, VariableNoiseKalmanFilter
from .fusion import (
    FusedPipelineResult,
    FusedSegmentResult,
    FusedTrajectoryResult,
    FusionState,
    FusionStep,
    SensorFuser,
)
from .pipeline import (
    GPSKalmanPipeline,
    PipelineResult,
    SegmentResult,
    SensorFusionPipeline,
    TrajectorySegment,
    TrajectorySplitter,
)
from .sensors import (
    CaptureRecaptureConfig,
    CaptureRecaptureMeasurement,
    CellTowerConfig,
    CellTowerMeasurement,
    CustomMeasurement,
    CustomSensorConfig,
    GPSConfig,
    GPSMeasurement,
    Measurement,
    SensorConfig,
)
from .state import ConstantVelocity2D, StateSpaceModel

__all__ = [
    # Single-sensor pipeline
    "GPSKalmanPipeline",
    "PipelineResult",
    "SegmentResult",
    "TrajectorySplitter",
    "TrajectorySegment",
    # Multi-sensor fusion pipeline
    "SensorFusionPipeline",
    # Sensors
    "GPSConfig",
    "GPSMeasurement",
    "CaptureRecaptureConfig",
    "CaptureRecaptureMeasurement",
    "CellTowerConfig",
    "CellTowerMeasurement",
    "CustomSensorConfig",
    "CustomMeasurement",
    "Measurement",
    "SensorConfig",
    # State models
    "StateSpaceModel",
    "ConstantVelocity2D",
    # Filters
    "VariableNoiseKalmanFilter",
    "KalmanFilterResult",
    # Fusion engine + results
    "SensorFuser",
    "FusionState",
    "FusionStep",
    "FusedTrajectoryResult",
    "FusedSegmentResult",
    "FusedPipelineResult",
]

__version__ = "0.2.0"
