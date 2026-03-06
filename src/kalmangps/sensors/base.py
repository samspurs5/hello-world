"""Abstract base classes for sensor measurements.

Designed for extensibility — adding new sensors (IMU, barometer, WiFi, etc.)
only requires subclassing ``Measurement`` and ``Sensor``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd


@dataclass
class Measurement(ABC):
    """A single observation from one sensor at one point in time.

    Subclasses must declare:
      - ``sensor_type``: a class-level string tag (e.g. ``"gps"``, ``"imu"``).
      - ``values``:      the observation vector (shape ``(n_obs,)``).
      - ``covariance``:  the observation noise covariance (shape ``(n_obs, n_obs)``).

    The covariance matrix encodes how uncertain this particular measurement is.
    For GPS with a reported horizontal accuracy this maps directly to the R
    matrix in the Kalman update step, enabling per-measurement noise.
    """

    sensor_type: ClassVar[str]

    timestamp: pd.Timestamp
    values: np.ndarray      # shape (n_obs,)
    covariance: np.ndarray  # shape (n_obs, n_obs)

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Dimensionality of the observation vector."""

    def validate(self) -> None:
        """Raise ``ValueError`` if the measurement is internally inconsistent."""
        if self.values.shape != (self.n_obs,):
            raise ValueError(
                f"Expected values shape ({self.n_obs},), got {self.values.shape}"
            )
        if self.covariance.shape != (self.n_obs, self.n_obs):
            raise ValueError(
                f"Expected covariance shape ({self.n_obs}, {self.n_obs}), "
                f"got {self.covariance.shape}"
            )


@dataclass
class SensorConfig(ABC):
    """Configuration / calibration for a sensor type.

    Holds the *observation model*: the matrix H that projects from the shared
    state vector to what this sensor observes, plus any sensor-specific
    hyper-parameters.

    Multi-sensor fusion wires together multiple ``SensorConfig`` objects so
    that each can contribute updates to a shared state.
    """

    #: Human-readable name used for logging / diagnostics.
    name: str = field(default="sensor")

    @property
    @abstractmethod
    def observation_matrix(self) -> np.ndarray:
        """H matrix: maps state vector -> observation space.

        Shape: ``(n_obs, n_state)``.
        """

    @abstractmethod
    def measurement_from_row(self, row: pd.Series) -> Measurement:
        """Convert a DataFrame row into a ``Measurement`` for this sensor."""
