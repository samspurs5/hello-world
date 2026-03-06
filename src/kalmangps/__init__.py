"""kalmangps — Kalman filter pipeline for GPS trajectories with multi-sensor fusion.

Quick start — single-sensor GPS
--------------------------------
>>> import pandas as pd
>>> from kalmangps import GPSKalmanPipeline, GPSConfig
>>>
>>> pipeline = GPSKalmanPipeline(
...     gps_config=GPSConfig(lat_col="lat", lon_col="lon",
...                          accuracy_col="h_accuracy",
...                          timestamp_col="utc_time"),
...     group_col="trip_id",
...     max_time_gap="5 min",
... )
>>> result = pipeline.run(df)
>>> smoothed = result.to_dataframe()

Quick start — multi-sensor fusion
-----------------------------------
>>> from kalmangps import SensorFusionPipeline, GPSConfig, SensorFactory
>>>
>>> pipeline = SensorFusionPipeline(
...     sensors={
...         "gps": GPSConfig(),
...         "entrance_cam": SensorFactory.fixed_point(
...             lat=51.5074, lon=-0.1278, uncertainty_m=20.0,
...             confidence_col="score",
...         ),
...         "tower_A": SensorFactory.range_sensor(
...             lat=51.501, lon=-0.090, range_std_m=300.0,
...         ),
...     },
...     primary_sensor="gps",
...     group_col="trip_id",
... )
>>> result = pipeline.run(primary_df=gps_df,
...                       secondary_dfs={"entrance_cam": cam_df, "tower_A": cell_df})
>>> df = result.to_dataframe()
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
    BearingModel,
    CallableModel,
    FixedPointModel,
    GenericMeasurement,
    GenericSensorConfig,
    GPSConfig,
    GPSMeasurement,
    Measurement,
    ObservationModel,
    RangeModel,
    SensorConfig,
    SensorFactory,
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
    # GPS (primary sensor)
    "GPSConfig",
    "GPSMeasurement",
    # Secondary sensors — use SensorFactory to create them
    "SensorFactory",
    "GenericSensorConfig",
    "GenericMeasurement",
    # Observation models (composable math)
    "ObservationModel",
    "FixedPointModel",
    "RangeModel",
    "BearingModel",
    "CallableModel",
    # Core abstractions
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

__version__ = "0.3.0"
