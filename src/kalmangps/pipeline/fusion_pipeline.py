"""Multi-sensor fusion pipeline.

``SensorFusionPipeline`` orchestrates the full workflow for fusing GPS data
with any combination of secondary sensors (cell towers, capture-recapture
readers, custom sensors):

1. **Split** the primary GPS DataFrame by a user column, then by time gaps.
2. **Window** each secondary sensor's DataFrame to the time bounds of each
   segment.
3. **Merge** all sensor measurements into a single time-ordered stream.
4. **Call** ``SensorFuser.run()`` on each segment stream.
5. **Return** a ``FusedPipelineResult`` containing per-segment results and
   a ``to_dataframe()`` convenience method.

Reference coordinates
---------------------
Each segment gets its own local Cartesian reference point (derived from the
first GPS fix in that segment).  Before processing a segment, the pipeline
calls ``sensor_config.set_reference(lat_ref, lon_ref)`` on every sensor so
their internal Cartesian coordinates are aligned with the GPS segment's origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..fusion.base import SensorFuser
from ..fusion.result import FusedPipelineResult, FusedSegmentResult, FusedTrajectoryResult
from ..sensors.base import Measurement, SensorConfig
from ..sensors.gps import GPSConfig
from ..state.models import ConstantVelocity2D, StateSpaceModel
from .splitter import TrajectorySplitter


class SensorFusionPipeline:
    """Multi-sensor fusion pipeline.

    Parameters
    ----------
    sensors:
        Dict mapping a name to a ``SensorConfig``.  The primary GPS sensor
        should be included here (key ``"gps"`` by convention) alongside any
        secondary sensors.
    primary_sensor:
        Key in *sensors* that provides the GPS-style position observations used
        to define trajectory segments (splitting by column and time gap).
    group_col:
        Column in the primary sensor DataFrame to split on (e.g. ``"trip_id"``).
    max_time_gap:
        Time gap threshold for splitting within a group.
    min_segment_length:
        Minimum number of primary-sensor fixes required to process a segment.
    state_model:
        State-space model shared by all sensors.

    Usage
    -----
    .. code-block:: python

        pipeline = SensorFusionPipeline(
            sensors={
                "gps":    GPSConfig(lat_col="lat", lon_col="lon",
                                    accuracy_col="hacc", timestamp_col="t"),
                "tower1": CellTowerConfig(tower_lat=51.5, tower_lon=-0.1,
                                          measurement_type="range",
                                          timestamp_col="t"),
                "cam1":   CaptureRecaptureConfig(reader_lat=51.51,
                                                  reader_lon=-0.09,
                                                  detection_radius_m=30,
                                                  timestamp_col="t"),
            },
            primary_sensor="gps",
            group_col="trip_id",
            max_time_gap="5 min",
        )

        result = pipeline.run(
            primary_df=gps_df,
            secondary_dfs={"tower1": cell_df, "cam1": camera_df},
        )
        df = result.to_dataframe()
    """

    def __init__(
        self,
        sensors: dict[str, SensorConfig],
        primary_sensor: str = "gps",
        group_col: str | None = None,
        max_time_gap: pd.Timedelta | str | int = pd.Timedelta("5 min"),
        min_segment_length: int = 3,
        state_model: StateSpaceModel | None = None,
    ) -> None:
        if primary_sensor not in sensors:
            raise ValueError(
                f"primary_sensor='{primary_sensor}' not found in sensors dict. "
                f"Available: {list(sensors.keys())}"
            )
        self.sensors = sensors
        self.primary_sensor = primary_sensor
        self.state_model = state_model or ConstantVelocity2D()
        self.fuser = SensorFuser(state_model=self.state_model)

        primary_cfg = sensors[primary_sensor]
        timestamp_col = getattr(primary_cfg, "timestamp_col", "timestamp")

        self.splitter = TrajectorySplitter(
            group_col=group_col,
            timestamp_col=timestamp_col,
            max_time_gap=max_time_gap,
            min_segment_length=min_segment_length,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        primary_df: pd.DataFrame,
        secondary_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> FusedPipelineResult:
        """Run the full fusion pipeline.

        Parameters
        ----------
        primary_df:
            DataFrame for the primary (GPS) sensor.
        secondary_dfs:
            Dict mapping sensor name -> DataFrame for each secondary sensor.
            Keys must appear in ``self.sensors``.  Missing sensors are silently
            skipped.

        Returns
        -------
        :class:`~kalmangps.fusion.result.FusedPipelineResult`
        """
        secondary_dfs = secondary_dfs or {}
        result = FusedPipelineResult()

        # Split primary sensor data into segments
        segments = self.splitter.split(primary_df)

        primary_cfg_template = self.sensors[self.primary_sensor]

        for segment in segments:
            seg_df = segment.data
            ts_col = getattr(primary_cfg_template, "timestamp_col", "timestamp")

            # Determine segment time window
            seg_start = seg_df[ts_col].min()
            seg_end = seg_df[ts_col].max()

            # Build a fresh primary config per segment (to reset lat_ref/lon_ref)
            import copy
            primary_cfg = copy.copy(primary_cfg_template)
            primary_cfg.lat_ref = None  # type: ignore[attr-defined]
            primary_cfg.lon_ref = None  # type: ignore[attr-defined]

            # Convert primary sensor observations (sets lat_ref/lon_ref)
            if hasattr(primary_cfg, "measurements_from_df"):
                if hasattr(primary_cfg, "lat_ref"):
                    primary_measurements = primary_cfg.measurements_from_df(
                        seg_df, auto_ref=True
                    )
                else:
                    primary_measurements = primary_cfg.measurements_from_df(seg_df)
            else:
                primary_measurements = [
                    primary_cfg.measurement_from_row(row)
                    for _, row in seg_df.iterrows()
                ]

            if len(primary_measurements) < 2:
                continue

            lat_ref = getattr(primary_cfg, "lat_ref", 0.0) or 0.0
            lon_ref = getattr(primary_cfg, "lon_ref", 0.0) or 0.0

            # Build fused stream: start with primary sensor
            stream: list[tuple[Measurement, SensorConfig]] = [
                (m, primary_cfg) for m in primary_measurements
            ]

            # Add secondary sensors windowed to this segment's time range
            for name, cfg_template in self.sensors.items():
                if name == self.primary_sensor:
                    continue
                if name not in secondary_dfs:
                    continue

                sec_df = secondary_dfs[name]
                sec_ts_col = getattr(cfg_template, "timestamp_col", "timestamp")

                if sec_ts_col not in sec_df.columns:
                    continue

                sec_df = sec_df.copy()
                sec_df[sec_ts_col] = pd.to_datetime(sec_df[sec_ts_col], utc=True)

                # Window to segment time range (inclusive with a 1-second buffer)
                mask = (sec_df[sec_ts_col] >= seg_start) & (
                    sec_df[sec_ts_col] <= seg_end
                )
                windowed = sec_df[mask]

                if windowed.empty:
                    continue

                sec_cfg = copy.copy(cfg_template)
                sec_cfg.set_reference(lat_ref, lon_ref)

                for _, row in windowed.iterrows():
                    try:
                        meas = sec_cfg.measurement_from_row(row)
                        stream.append((meas, sec_cfg))
                    except Exception:
                        continue

            fused = self.fuser.run(
                stream,
                lat_ref=lat_ref,
                lon_ref=lon_ref,
            )

            result.segment_results.append(
                FusedSegmentResult(
                    group_key=segment.group_key,
                    segment_index=segment.segment_index,
                    fused=fused,
                )
            )

        return result
