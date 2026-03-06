"""Multi-sensor fusion layer — prepared for the next implementation phase.

Architecture overview
---------------------
Multi-sensor fusion (e.g. GPS + IMU + barometer) works by maintaining a
*single shared state* and applying one Kalman update per sensor reading
that arrives at each time step.

The fusion layer owns the predict step and coordinates the update steps:

    for each time step:
        fuser.predict(dt)
        for sensor, measurement in available_measurements:
            fuser.update(measurement, sensor.observation_matrix)

This module defines:

- ``FusionState`` — the mutable filter state (x, P) shared across sensors.
- ``SensorFuser`` — orchestrates predict/update for multiple sensors.
- ``FusedTrajectoryResult`` — output container.

Only ``FusionState`` and ``SensorFuser`` are currently implemented as stubs
with full docstrings so the design is clear and adding sensors later requires
only:

1. Subclassing ``SensorConfig`` (already done in ``sensors/``).
2. Registering it with ``SensorFuser.register_sensor()``.
3. Feeding its measurements into ``SensorFuser.run()``.

No changes to the state model or Kalman update math are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..filters.kalman import VariableNoiseKalmanFilter
from ..sensors.base import Measurement, SensorConfig
from ..state.models import StateSpaceModel


@dataclass
class FusionState:
    """Mutable Kalman filter state shared by all sensors.

    Parameters
    ----------
    x:
        State vector, shape ``(n_state,)``.
    P:
        State covariance matrix, shape ``(n_state, n_state)``.
    timestamp:
        Time corresponding to the current state estimate.
    """

    x: np.ndarray
    P: np.ndarray
    timestamp: pd.Timestamp

    def predict(self, dt: float, state_model: StateSpaceModel) -> None:
        """Advance the state estimate by *dt* seconds in-place."""
        F = state_model.transition_matrix(dt)
        Q = state_model.process_noise(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> np.ndarray:
        """Apply a single sensor observation in-place.

        Parameters
        ----------
        z:
            Observation vector, shape ``(n_obs,)``.
        H:
            Observation matrix, shape ``(n_obs, n_state)``.
        R:
            Observation noise covariance, shape ``(n_obs, n_obs)``.

        Returns
        -------
        innovation:
            The pre-update residual ``z - H @ x``, shape ``(n_obs,)``.
        """
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        return y


class SensorFuser:
    """Multi-sensor Kalman fusion orchestrator.

    Maintains a single ``FusionState`` and processes a time-ordered stream of
    measurements from heterogeneous sensors.

    Parameters
    ----------
    state_model:
        Shared state-space model (F, Q).
    sensors:
        Mapping from sensor name -> ``SensorConfig``.  Each config provides
        the observation matrix H for that sensor.

    Usage (future)
    --------------
    >>> fuser = SensorFuser(
    ...     state_model=ConstantVelocity2D(process_noise_std=0.5),
    ...     sensors={"gps": GPSConfig(...), "imu": IMUConfig(...)},
    ... )
    >>> result = fuser.run(measurements_stream)  # list of (Measurement, SensorConfig)
    """

    def __init__(
        self,
        state_model: StateSpaceModel,
        sensors: dict[str, SensorConfig] | None = None,
    ) -> None:
        self.state_model = state_model
        self.sensors: dict[str, SensorConfig] = sensors or {}

    def register_sensor(self, name: str, config: SensorConfig) -> None:
        """Add or replace a sensor registration."""
        self.sensors[name] = config

    def run(
        self,
        measurements: list[tuple[Measurement, SensorConfig]],
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> list[FusionState]:
        """Process a time-ordered stream of (Measurement, SensorConfig) pairs.

        Each predict step spans the gap to the *next* measurement; each update
        step applies that measurement.  When two sensors fire at the same
        timestamp they share the predict step (dt=0 for the second).

        Parameters
        ----------
        measurements:
            Time-ordered list of ``(Measurement, SensorConfig)`` pairs.
        x0, P0:
            Optional override for initial state.

        Returns
        -------
        List of ``FusionState`` snapshots after each update step.

        .. note::
            This method is a *stub* — the full implementation is coming in the
            next phase.  The interface is intentionally final so callers can
            be written now.
        """
        raise NotImplementedError(
            "Multi-sensor fusion is coming in the next implementation phase. "
            "Use GPSKalmanPipeline for single-sensor GPS filtering today."
        )
