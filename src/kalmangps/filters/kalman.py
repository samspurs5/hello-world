"""Kalman filter implementation with per-measurement noise support.

Design rationale
----------------
``pykalman``'s ``KalmanFilter`` class supports EM-based parameter estimation
(learning F, Q, H, R from data) but applies a *fixed* observation covariance R
at every time step.  GPS data carries a per-fix accuracy estimate that we want
to respect — a fix with 3 m accuracy should be trusted much more than one with
50 m accuracy.

This module provides:

1. ``KalmanFilterResult`` — a plain dataclass holding all filter outputs.
2. ``VariableNoiseKalmanFilter`` — implements the standard Kalman
   predict/update cycle manually so that a distinct R matrix can be applied at
   every step.  ``pykalman`` is used optionally for EM-based initialisation of
   the process-noise parameter ``Q`` when ``fit_process_noise=True``.

The filter is designed to accept *any* ``SensorConfig`` (not just GPS), which
makes it the natural integration point for multi-sensor fusion: the fusion
layer will call the predict step once and the update step once per sensor that
has a reading at that time.
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
    """Outputs of a Kalman filter run on one trajectory segment.

    All arrays are aligned with the input measurements (one row per fix).

    Attributes
    ----------
    timestamps:
        UTC timestamps of each observation.
    state_means:
        Filtered state vector at each step, shape ``(n_steps, n_state)``.
    state_covariances:
        Filtered state covariance at each step, shape
        ``(n_steps, n_state, n_state)``.
    innovations:
        Observation residuals ``z - H·x̂``, shape ``(n_steps, n_obs)``.
    innovation_covariances:
        Innovation covariance ``S = H·P·Hᵀ + R``, shape
        ``(n_steps, n_obs, n_obs)``.
    sensor_type:
        Identifier of the sensor that produced the observations.
    """

    timestamps: list[pd.Timestamp]
    state_means: np.ndarray
    state_covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    sensor_type: str = "unknown"

    # Convenience ------------------------------------------------------------ #

    def position_xy(self) -> np.ndarray:
        """Return the 2-D position columns ``[x, y]``, shape ``(n, 2)``."""
        return self.state_means[:, :2]

    def velocity_xy(self) -> np.ndarray:
        """Return the 2-D velocity columns ``[vx, vy]``, shape ``(n, 2)``."""
        return self.state_means[:, 2:4]

    def position_uncertainty(self) -> np.ndarray:
        """1-sigma position uncertainty (m) at each step, shape ``(n, 2)``."""
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
    """Kalman filter that supports per-measurement observation covariance.

    Parameters
    ----------
    state_model:
        The state-space model (transition matrix F and process noise Q).
        Defaults to :class:`~kalmangps.state.ConstantVelocity2D`.
    fit_process_noise:
        When ``True``, run ``pykalman``'s EM algorithm on the first segment
        seen to refine ``state_model.process_noise_std``.  Useful when the
        dynamics are unknown.  Defaults to ``False``.
    """

    def __init__(
        self,
        state_model: StateSpaceModel | None = None,
        fit_process_noise: bool = False,
    ) -> None:
        self.state_model: StateSpaceModel = state_model or ConstantVelocity2D()
        self.fit_process_noise = fit_process_noise
        self._em_fitted = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def filter(
        self,
        measurements: list[Measurement],
        sensor_config: SensorConfig,
    ) -> KalmanFilterResult:
        """Run the Kalman filter over a list of measurements.

        Parameters
        ----------
        measurements:
            Ordered (by time) list of ``Measurement`` objects.
        sensor_config:
            The sensor configuration providing the observation matrix H.

        Returns
        -------
        KalmanFilterResult
        """
        if len(measurements) == 0:
            raise ValueError("measurements must not be empty")

        H = sensor_config.observation_matrix  # (n_obs, n_state)
        n_state = self.state_model.n_state
        n_obs = H.shape[0]
        n = len(measurements)

        # --- optional EM initialisation ------------------------------------ #
        if self.fit_process_noise and not self._em_fitted:
            self._em_fit(measurements, H)

        # --- allocate output arrays ---------------------------------------- #
        state_means = np.zeros((n, n_state))
        state_covs = np.zeros((n, n_state, n_state))
        innovations = np.zeros((n, n_obs))
        innov_covs = np.zeros((n, n_obs, n_obs))

        # --- initial state -------------------------------------------------- #
        x, P = self.state_model.initial_state(measurements[0].values)

        # --- filter loop ---------------------------------------------------- #
        prev_ts = measurements[0].timestamp
        for i, meas in enumerate(measurements):
            dt = (meas.timestamp - prev_ts).total_seconds() if i > 0 else 0.0
            dt = max(dt, 1e-6)  # guard against duplicate timestamps

            # Predict
            if i > 0:
                F = self.state_model.transition_matrix(dt)
                Q = self.state_model.process_noise(dt)
                x = F @ x
                P = F @ P @ F.T + Q

            # Update
            R = meas.covariance  # per-measurement noise (n_obs, n_obs)
            x, P, y, S = self._update(x, P, meas.values, H, R)

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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _update(
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Standard Kalman update step.

        Returns updated ``(x, P, innovation, innovation_covariance)``.
        """
        y = z - H @ x                  # innovation
        S = H @ P @ H.T + R            # innovation covariance
        K = P @ H.T @ np.linalg.inv(S) # Kalman gain
        x_new = x + K @ y
        I = np.eye(len(x))
        P_new = (I - K @ H) @ P        # Joseph form would be more stable but this is sufficient
        return x_new, P_new, y, S

    def _em_fit(self, measurements: list[Measurement], H: np.ndarray) -> None:
        """Use pykalman EM to estimate a good process_noise_std."""
        try:
            from pykalman import KalmanFilter as PyKalmanFilter
        except ImportError:
            return

        obs = np.stack([m.values for m in measurements])
        # Use a simple scalar R (average accuracy) for EM
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
            # Back out a scalar process_noise_std from the fitted Q diagonal
            q_diag = np.diag(kf.transition_covariance)
            std_estimate = float(np.sqrt(np.mean(q_diag)))
            if isinstance(self.state_model, ConstantVelocity2D) and std_estimate > 0:
                self.state_model.process_noise_std = std_estimate
        except Exception:
            pass  # fall back to user-supplied value silently

        self._em_fitted = True
