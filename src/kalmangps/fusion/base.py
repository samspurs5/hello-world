"""Multi-sensor Extended Kalman Filter fusion engine.

Architecture
------------
``SensorFuser`` maintains a single shared state ``[x, y, vx, vy]`` and
processes a time-ordered stream of measurements from heterogeneous sensors.

The EKF update step is::

    H   = sensor_config.observation_jacobian(x̂)   # linearise at current estimate
    z̃   = sensor_config.observe(x̂)                # predicted measurement
    y   = z − z̃                                   # innovation
    S   = H·P·Hᵀ + R                              # innovation covariance
    K   = P·Hᵀ·S⁻¹                                # Kalman gain
    x̂   = x̂ + K·y
    P   = (I − K·H)·P

For **linear** sensors (GPS, capture-recapture), ``observation_jacobian``
returns a constant H and ``observe`` returns ``H @ state``, so the EKF
reduces to the standard KF exactly.

For **nonlinear** sensors (cell-tower range/angle), the Jacobian is evaluated
at the current state estimate and the update proceeds identically.

When multiple sensors fire at the same timestamp, the predict step runs once
and then each sensor's update is applied sequentially (equivalent to a joint
update for zero-correlation sensors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
import pandas as pd

from ..sensors.base import Measurement, SensorConfig
from ..state.models import ConstantVelocity2D, StateSpaceModel
from .result import FusedTrajectoryResult, FusionStep


@dataclass
class FusionState:
    """Mutable Kalman filter state shared by all sensors.

    Attributes
    ----------
    x:
        State vector ``[x, y, vx, vy]``, shape ``(n_state,)``.
    P:
        State covariance matrix, shape ``(n_state, n_state)``.
    timestamp:
        Time of the most recent update.
    """

    x: np.ndarray
    P: np.ndarray
    timestamp: pd.Timestamp

    def predict(self, dt: float, state_model: StateSpaceModel) -> None:
        """Advance the state estimate by *dt* seconds in-place."""
        if dt <= 0:
            return
        F = state_model.transition_matrix(dt)
        Q = state_model.process_noise(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        z: np.ndarray,
        sensor_config: SensorConfig,
        R: np.ndarray,
    ) -> np.ndarray:
        """Apply one EKF measurement update in-place.

        Parameters
        ----------
        z:
            Observation vector, shape ``(n_obs,)``.
        sensor_config:
            Provides ``observe()`` and ``observation_jacobian()`` evaluated at
            the current state estimate ``self.x``.
        R:
            Per-measurement observation noise covariance, shape
            ``(n_obs, n_obs)``.

        Returns
        -------
        innovation:
            Pre-update residual ``z − h(x̂)``, shape ``(n_obs,)``.
        """
        H = sensor_config.observation_jacobian(self.x)   # linearise
        z_pred = sensor_config.observe(self.x)            # predicted obs
        y = z - z_pred                                    # innovation

        # Wrap angle innovations into (−π, π] for bearing sensors
        if z.shape == (1,) and np.abs(y[0]) > np.pi:
            y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P

        return y


class SensorFuser:
    """Multi-sensor EKF fusion engine.

    Parameters
    ----------
    state_model:
        Shared state-space model providing ``transition_matrix(dt)`` and
        ``process_noise(dt)``.
    """

    def __init__(self, state_model: StateSpaceModel | None = None) -> None:
        self.state_model: StateSpaceModel = state_model or ConstantVelocity2D()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        measurements: list[tuple[Measurement, SensorConfig]],
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        lat_ref: float = 0.0,
        lon_ref: float = 0.0,
    ) -> FusedTrajectoryResult:
        """Fuse a time-ordered stream of (Measurement, SensorConfig) pairs.

        Parameters
        ----------
        measurements:
            Each item is ``(measurement, sensor_config)``.  The list must be
            sorted by ``measurement.timestamp``.
        x0:
            Initial state vector.  Defaults to the first measurement projected
            into state space with zero velocity.
        P0:
            Initial state covariance.  Defaults to large diagonal (high
            uncertainty).
        lat_ref, lon_ref:
            Reference point stored in the result for lat/lon back-projection.

        Returns
        -------
        :class:`~kalmangps.fusion.result.FusedTrajectoryResult`
        """
        if not measurements:
            return FusedTrajectoryResult(lat_ref=lat_ref, lon_ref=lon_ref)

        # Sort defensively
        measurements = sorted(measurements, key=lambda t: t[0].timestamp)

        # Initialise state
        first_meas, _ = measurements[0]
        if x0 is None or P0 is None:
            x0_default, P0_default = self.state_model.initial_state(first_meas.values)
            x0 = x0 if x0 is not None else x0_default
            P0 = P0 if P0 is not None else P0_default

        state = FusionState(x=x0.copy(), P=P0.copy(), timestamp=first_meas.timestamp)
        steps: list[FusionStep] = []

        # Group by timestamp so sensors firing simultaneously share one predict
        for ts, group_iter in groupby(measurements, key=lambda t: t[0].timestamp):
            group = list(group_iter)

            # Predict to this timestamp
            dt = (ts - state.timestamp).total_seconds()
            state.predict(dt, self.state_model)

            # Sequential EKF update for each sensor at this timestamp
            for meas, cfg in group:
                innovation = state.update(meas.values, cfg, meas.covariance)
                steps.append(
                    FusionStep(
                        timestamp=ts,
                        state_mean=state.x.copy(),
                        state_cov=state.P.copy(),
                        sensor_type=meas.sensor_type,
                        innovation=innovation,
                    )
                )

            state.timestamp = ts

        return FusedTrajectoryResult(steps=steps, lat_ref=lat_ref, lon_ref=lon_ref)
