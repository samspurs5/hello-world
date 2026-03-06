"""Main pipeline: split → filter → collect results.

``GPSKalmanPipeline`` is the primary user-facing entry point.  A typical
workflow:

    >>> import pandas as pd
    >>> from kalmangps import GPSKalmanPipeline, GPSConfig
    >>>
    >>> df = pd.read_csv("gps_log.csv")
    >>> pipeline = GPSKalmanPipeline(
    ...     gps_config=GPSConfig(lat_col="lat", lon_col="lon",
    ...                          accuracy_col="h_accuracy",
    ...                          timestamp_col="time"),
    ...     group_col="trip_id",
    ...     max_time_gap="5 min",
    ... )
    >>> results = pipeline.run(df)
    >>> smoothed_df = results.to_dataframe()

The ``PipelineResult`` container exposes per-segment ``KalmanFilterResult``
objects and provides helpers to reassemble everything into a flat DataFrame
annotated with filtered lat/lon, velocity, and uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..filters.kalman import KalmanFilterResult, VariableNoiseKalmanFilter
from ..sensors.gps import GPSConfig, xy_to_latlon
from ..state.models import ConstantVelocity2D, StateSpaceModel
from .splitter import TrajectorySegment, TrajectorySplitter


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class SegmentResult:
    """Kalman filter output for one trajectory segment, plus provenance."""

    segment: TrajectorySegment
    filter_result: KalmanFilterResult

    def to_dataframe(self, lat_ref: float, lon_ref: float) -> pd.DataFrame:
        """Build a flat DataFrame with filtered coordinates for this segment.

        Columns added (on top of the original segment data):
          ``filtered_lat``, ``filtered_lon``,
          ``filtered_vx_ms``, ``filtered_vy_ms``,
          ``pos_uncertainty_x_m``, ``pos_uncertainty_y_m``,
          ``segment_index``, ``group_key``.
        """
        fr = self.filter_result
        xy = fr.position_xy()
        vel = fr.velocity_xy()
        unc = fr.position_uncertainty()

        lats, lons = xy_to_latlon(xy[:, 0], xy[:, 1], lat_ref, lon_ref)

        out = self.segment.data.copy()
        out["filtered_lat"] = lats
        out["filtered_lon"] = lons
        out["filtered_vx_ms"] = vel[:, 0]
        out["filtered_vy_ms"] = vel[:, 1]
        out["pos_uncertainty_x_m"] = unc[:, 0]
        out["pos_uncertainty_y_m"] = unc[:, 1]
        out["segment_index"] = self.segment.segment_index
        out["group_key"] = str(self.segment.group_key)
        return out


@dataclass
class PipelineResult:
    """Aggregated results from a full pipeline run."""

    segment_results: list[SegmentResult] = field(default_factory=list)
    #: Mapping from segment -> reference (lat, lon) used for that segment.
    reference_points: dict[int, tuple[float, float]] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate all segment results into one DataFrame."""
        if not self.segment_results:
            return pd.DataFrame()

        frames = []
        for i, sr in enumerate(self.segment_results):
            lat_ref, lon_ref = self.reference_points.get(i, (0.0, 0.0))
            frames.append(sr.to_dataframe(lat_ref, lon_ref))

        return pd.concat(frames, ignore_index=True)

    @property
    def n_segments(self) -> int:
        return len(self.segment_results)

    @property
    def n_points(self) -> int:
        return sum(sr.segment.n_points for sr in self.segment_results)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class GPSKalmanPipeline:
    """End-to-end pipeline: split a GPS DataFrame, Kalman-filter each segment.

    Parameters
    ----------
    gps_config:
        Describes the GPS sensor columns and default accuracy.
    group_col:
        Column used to split the data into independent trajectories (e.g.
        ``"device_id"``, ``"trip_id"``).  ``None`` = treat all rows as one
        trajectory.
    max_time_gap:
        Time gap threshold for splitting within a group.  Accepts any value
        that ``pd.Timedelta`` accepts (``"5 min"``, ``300``, etc.).
    min_segment_length:
        Minimum number of GPS fixes required to filter a segment.  Shorter
        segments are silently dropped.
    state_model:
        The state-space model.  Defaults to :class:`ConstantVelocity2D`.
    fit_process_noise:
        Run pykalman EM on the first segment to tune process noise
        automatically.  Useful when dynamics are unknown.
    """

    def __init__(
        self,
        gps_config: GPSConfig | None = None,
        group_col: str | None = None,
        max_time_gap: pd.Timedelta | str | int = pd.Timedelta("5 min"),
        min_segment_length: int = 3,
        state_model: StateSpaceModel | None = None,
        fit_process_noise: bool = False,
    ) -> None:
        self.gps_config = gps_config or GPSConfig()
        self.state_model = state_model or ConstantVelocity2D()
        self.splitter = TrajectorySplitter(
            group_col=group_col,
            timestamp_col=self.gps_config.timestamp_col,
            max_time_gap=max_time_gap,
            min_segment_length=min_segment_length,
        )
        self.kf = VariableNoiseKalmanFilter(
            state_model=self.state_model,
            fit_process_noise=fit_process_noise,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """Run the full pipeline on *df*.

        Parameters
        ----------
        df:
            Raw GPS DataFrame.

        Returns
        -------
        :class:`PipelineResult`
        """
        segments = self.splitter.split(df)
        result = PipelineResult()

        for i, segment in enumerate(segments):
            # Each segment gets its own reference point for numerical stability.
            cfg = GPSConfig(
                name=self.gps_config.name,
                lat_col=self.gps_config.lat_col,
                lon_col=self.gps_config.lon_col,
                accuracy_col=self.gps_config.accuracy_col,
                timestamp_col=self.gps_config.timestamp_col,
                default_accuracy_m=self.gps_config.default_accuracy_m,
                lat_ref=None,  # will be set from first row of segment
                lon_ref=None,
            )

            measurements = cfg.measurements_from_df(segment.data, auto_ref=True)
            if len(measurements) < 2:
                continue

            fr = self.kf.filter(measurements, cfg)
            result.segment_results.append(SegmentResult(segment=segment, filter_result=fr))
            result.reference_points[i] = (cfg.lat_ref, cfg.lon_ref)  # type: ignore[assignment]

        return result
