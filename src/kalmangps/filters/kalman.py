"""Kalman / Extended Kalman filter with per-measurement noise support.

The filter now uses the EKF interface (``sensor_config.observe()`` and
``sensor_config.observation_jacobian()``) so it handles both linear sensors
(GPS, capture-recapture) and nonlinear sensors (cell-tower range/angle)
identically.  For linear sensors the EKF update is mathematically identical
to the standard KF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..sensors.base import Measurement, SensorConfig
from ..state.models import ConstantVelocity2D, StateSpaceModel


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class KalmanFilterResult:
    """Outputs of a Kalman filter run on one trajectory segment."""

    timestamps: list[pd.Timestamp]
    state_means: np.ndarray
    state_covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    sensor_type: str = "unknown"

    def position_xy(self) -> np.ndarray:
        return self.state_means[:, :2]

    def velocity_xy(self) -> np.ndarray:
        return self.state_means[:, 2:4]

    def position_uncertainty(self) -> np.ndarray:
        return np.sqrt(
            np.stack(
                [self.state_covariances[:, 0, 0],
                 self.state_covariances[:, 1, 1]],
                axis=1,
            )
        )


# --------------------------------------------------------------------------- #
# Core filter
# --------------------------------------------------------------------------- #

class VariableNoiseKalmanFilter:
    """EKF-based filter with per-measurement observation covariance.

    Supports both linear (GPS) and nonlinear (cell-tower) sensors via the
    uniform ``SensorConfig.observe()`` / ``observation_jacobian()`` interface.
    """

    def __init__(
        self,
        state_model: StateSpaceModel | None = None,
        fit_process_noise: bool = False,
    ) -> None:
        self.state_model: StateSpaceModel = state_model or ConstantVelocity2D()
        self.fit_process_noise = fit_process_noise
        self._em_fitted = False

    def filter(
        self,
        measurements: list[Measurement],
        sensor_config: SensorConfig,
    ) -> KalmanFilterResult:
        if len(measurements) == 0:
            raise ValueError("measurements must not be empty")

        n_state = self.state_model.n_state
        n_obs = measurements[0].values.shape[0]
        n = len(measurements)

        # Optional EM init (linear sensors only)
        H_const = sensor_config.observation_matrix
        if self.fit_process_noise and not self._em_fitted and H_const is not None:
            self._em_fit(measurements, H_const)

        state_means = np.zeros((n, n_state))
        state_covs = np.zeros((n, n_state, n_state))
        innovations = np.zeros((n, n_obs))
        innov_covs = np.zeros((n, n_obs, n_obs))

        x, P = self.state_model.initial_state(measurements[0].values)

        prev_ts = measurements[0].timestamp
        for i, meas in enumerate(measurements):
            dt = (meas.timestamp - prev_ts).total_seconds() if i > 0 else 0.0
            dt = max(dt, 1e-6)

            if i > 0:
                F = self.state_model.transition_matrix(dt)
                Q = self.state_model.process_noise(dt)
                x = F @ x
                P = F @ P @ F.T + Q

            R = meas.covariance
            x, P, y, S = self._ekf_update(x, P, meas.values, sensor_config, R)

            state_means[i] = x
            state_covs[i] = P
            innovations[i] = y
            innov_covs[i] = S
            prev_ts = meas.timestamp

        return KalmanFilterResult(
            timestamps=[m.timestamp for m in measurements],
            state_means=state_means,
            state_covariances=state_covs,
            innovations=innovations,
            innovation_covariances=innov_covs,
            sensor_type=measurements[0].sensor_type,
        )

    @staticmethod
    def _ekf_update(
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        sensor_config: SensorConfig,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """EKF update step (degenerates to KF for linear sensors)."""
        H = sensor_config.observation_jacobian(x)
        z_pred = sensor_config.observe(x)
        y = z - z_pred
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x_new = x + K @ y
        P_new = (np.eye(len(x)) - K @ H) @ P
        return x_new, P_new, y, S

    def _em_fit(self, measurements: list[Measurement], H: np.ndarray) -> None:
        try:
            from pykalman import KalmanFilter as PyKalmanFilter
        except ImportError:
            return

        obs = np.stack([m.values for m in measurements])
        avg_r = np.mean([m.covariance for m in measurements], axis=0)

        kf = PyKalmanFilter(
            transition_matrices=self.state_model.transition_matrix(1.0),
            observation_matrices=H,
            observation_covariance=avg_r,
            em_vars=["transition_covariance", "initial_state_mean",
                     "initial_state_covariance"],
            n_dim_state=self.state_model.n_state,
            n_dim_obs=H.shape[0],
        )
        try:
            kf = kf.em(obs, n_iter=10)
            q_diag = np.diag(kf.transition_covariance)
            std_estimate = float(np.sqrt(np.mean(q_diag)))
            if isinstance(self.state_model, ConstantVelocity2D) and std_estimate > 0:
                self.state_model.process_noise_std = std_estimate
        except Exception:
            pass

        self._em_fitted = True
