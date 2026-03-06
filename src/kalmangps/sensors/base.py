"""Abstract base classes for sensor measurements.

Designed for extensibility — adding new sensors (IMU, barometer, WiFi, etc.)
only requires subclassing ``Measurement`` and ``SensorConfig``.

EKF interface
-------------
The observation model is expressed through two methods on ``SensorConfig``:

- ``observe(state)``          — the observation function h(x)
- ``observation_jacobian(state)`` — ∂h/∂x evaluated at *state* (the H matrix)

For **linear** sensors (GPS, capture-recapture) these collapse to a constant
matrix and the Extended Kalman Filter (EKF) update is identical to the
standard KF update.  For **nonlinear** sensors (cell-tower ranging, AoA) the
nonlinear h and its Jacobian are provided explicitly.

This uniform interface means the same update code handles all sensor types.
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

    Observation model
    -----------------
    The EKF update only needs two things from each sensor:

    1. ``observe(state)`` — h(x): project the state into observation space.
    2. ``observation_jacobian(state)`` — ∂h/∂x at *state*.

    For linear sensors, subclasses may instead define ``observation_matrix``
    (the constant H) and inherit default implementations of the two methods
    above that call it.  For nonlinear sensors, override both methods directly.

    Reference coordinates
    ---------------------
    Sensors that express observations in a local Cartesian frame (metres from a
    reference point) should implement ``set_reference(lat_ref, lon_ref)`` to
    recompute their internal Cartesian coordinates when the fusion pipeline
    assigns a segment reference point.
    """

    #: Human-readable name used for logging / diagnostics.
    name: str = field(default="sensor")

    # ------------------------------------------------------------------ #
    # Observation model — override ONE of the two approaches below
    # ------------------------------------------------------------------ #

    @property
    def observation_matrix(self) -> np.ndarray | None:
        """Constant H matrix for linear sensors.  ``None`` for nonlinear ones."""
        return None

    def observe(self, state: np.ndarray) -> np.ndarray:
        """Observation function h(state) — predicted measurement given *state*.

        Default implementation multiplies ``observation_matrix @ state``.
        Nonlinear sensors must override this.
        """
        H = self.observation_matrix
        if H is not None:
            return H @ state
        raise NotImplementedError(
            f"{type(self).__name__} is nonlinear: implement observe()"
        )

    def observation_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian ∂h/∂x evaluated at *state*, shape ``(n_obs, n_state)``.

        Default implementation returns the constant ``observation_matrix``.
        Nonlinear sensors must override this.
        """
        H = self.observation_matrix
        if H is not None:
            return H
        raise NotImplementedError(
            f"{type(self).__name__} is nonlinear: implement observation_jacobian()"
        )

    # ------------------------------------------------------------------ #
    # Reference-point hook (coordinate projection)
    # ------------------------------------------------------------------ #

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        """Recompute any Cartesian coordinates when the reference point changes.

        Called by the fusion pipeline at the start of each segment so that all
        sensors share the same local origin.  The default is a no-op; sensors
        with fixed geographic locations (cell towers, camera readers) must
        override this.
        """

    # ------------------------------------------------------------------ #
    # Data ingestion
    # ------------------------------------------------------------------ #

    @abstractmethod
    def measurement_from_row(self, row: pd.Series) -> Measurement:
        """Convert a DataFrame row into a ``Measurement`` for this sensor."""

    def measurements_from_df(self, df: pd.DataFrame) -> list[Measurement]:
        """Convert an entire DataFrame into a list of measurements.

        Override for sensors that need batch context (e.g. auto reference
        setting).  The default calls ``measurement_from_row`` per row.
        """
        return [self.measurement_from_row(row) for _, row in df.iterrows()]
