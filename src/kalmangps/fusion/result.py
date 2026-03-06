"""Result containers for multi-sensor fusion output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..sensors.gps import xy_to_latlon


@dataclass
class FusionStep:
    """State snapshot produced by one Kalman update step.

    One ``FusionStep`` is recorded for each measurement processed, whether it
    came from GPS, a cell tower, a camera reader, or any other sensor.

    Attributes
    ----------
    timestamp:
        UTC time of the measurement that triggered this update.
    state_mean:
        Filtered state vector ``[x, y, vx, vy]`` (m, m/s), shape ``(n_state,)``.
    state_cov:
        State covariance matrix, shape ``(n_state, n_state)``.
    sensor_type:
        String tag of the sensor that contributed this update (e.g. ``"gps"``,
        ``"cell_tower"``, ``"capture_recapture"``).
    innovation:
        Pre-update residual ``z - h(x̂)``, shape ``(n_obs,)``.
    """

    timestamp: pd.Timestamp
    state_mean: np.ndarray
    state_cov: np.ndarray
    sensor_type: str
    innovation: np.ndarray

    def position_xy(self) -> np.ndarray:
        """Return ``[x, y]`` in metres, shape ``(2,)``."""
        return self.state_mean[:2]

    def velocity_xy(self) -> np.ndarray:
        """Return ``[vx, vy]`` in m/s, shape ``(2,)``."""
        return self.state_mean[2:4]

    def position_uncertainty(self) -> np.ndarray:
        """1-sigma position uncertainty ``[σx, σy]`` in metres, shape ``(2,)``."""
        return np.sqrt([self.state_cov[0, 0], self.state_cov[1, 1]])


@dataclass
class FusedTrajectoryResult:
    """All fusion steps for one trajectory segment.

    Parameters
    ----------
    steps:
        Time-ordered list of :class:`FusionStep` objects, one per sensor update.
    lat_ref, lon_ref:
        Reference coordinates used for the local Cartesian projection.
    """

    steps: list[FusionStep] = field(default_factory=list)
    lat_ref: float = 0.0
    lon_ref: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    # ------------------------------------------------------------------ #
    # Filtered-at-GPS-times convenience
    # ------------------------------------------------------------------ #

    def at_sensor(self, sensor_type: str) -> list[FusionStep]:
        """Return only the steps that were triggered by *sensor_type*."""
        return [s for s in self.steps if s.sensor_type == sensor_type]

    def sensor_types(self) -> set[str]:
        """Set of all sensor types that contributed updates."""
        return {s.sensor_type for s in self.steps}

    # ------------------------------------------------------------------ #
    # DataFrame export
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all fusion steps to a flat DataFrame.

        Columns
        -------
        ``timestamp``, ``sensor_type``,
        ``filtered_lat``, ``filtered_lon``,
        ``filtered_vx_ms``, ``filtered_vy_ms``,
        ``pos_uncertainty_x_m``, ``pos_uncertainty_y_m``,
        ``innovation_0`` [, ``innovation_1``] (one column per obs dimension).
        """
        if not self.steps:
            return pd.DataFrame()

        rows = []
        for step in self.steps:
            xy = step.position_xy()
            vxy = step.velocity_xy()
            unc = step.position_uncertainty()
            lat, lon = xy_to_latlon(xy[0], xy[1], self.lat_ref, self.lon_ref)
            row: dict = {
                "timestamp": step.timestamp,
                "sensor_type": step.sensor_type,
                "filtered_lat": float(lat),
                "filtered_lon": float(lon),
                "filtered_vx_ms": float(vxy[0]),
                "filtered_vy_ms": float(vxy[1]),
                "pos_uncertainty_x_m": float(unc[0]),
                "pos_uncertainty_y_m": float(unc[1]),
            }
            for j, iv in enumerate(step.innovation):
                row[f"innovation_{j}"] = float(iv)
            rows.append(row)

        return pd.DataFrame(rows)


@dataclass
class FusedSegmentResult:
    """Fused result for one trajectory segment with provenance metadata."""

    group_key: object
    segment_index: int
    fused: FusedTrajectoryResult

    def to_dataframe(self) -> pd.DataFrame:
        df = self.fused.to_dataframe()
        if df.empty:
            return df
        df["group_key"] = str(self.group_key)
        df["segment_index"] = self.segment_index
        return df


@dataclass
class FusedPipelineResult:
    """Aggregated fusion results across all trajectory segments."""

    segment_results: list[FusedSegmentResult] = field(default_factory=list)

    @property
    def n_segments(self) -> int:
        return len(self.segment_results)

    @property
    def n_steps(self) -> int:
        return sum(sr.fused.n_steps for sr in self.segment_results)

    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate all segment results into one DataFrame."""
        if not self.segment_results:
            return pd.DataFrame()
        frames = [sr.to_dataframe() for sr in self.segment_results]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
