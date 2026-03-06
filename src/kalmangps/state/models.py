"""State-space models for trajectory estimation.

A *state-space model* defines:
  - The **state vector** (what we are estimating).
  - The **transition matrix** F (how the state evolves over one time step).
  - The **process noise covariance** Q (how uncertain the dynamics are).

The current model is a 2-D **constant-velocity** model:

    state = [x, y, vx, vy]   (position + velocity in metres / m·s⁻¹)

Adding new models (constant-acceleration, constant-turn-rate, etc.) is
straightforward — subclass ``StateSpaceModel`` and implement the two abstract
methods.

For multi-sensor fusion the state model is shared; each sensor contributes an
observation matrix H (defined in its ``SensorConfig``) that maps from this
shared state into whatever the sensor actually measures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class StateSpaceModel(ABC):
    """Interface for a discrete-time linear state-space model."""

    @property
    @abstractmethod
    def n_state(self) -> int:
        """Dimension of the state vector."""

    @abstractmethod
    def transition_matrix(self, dt: float) -> np.ndarray:
        """Return F for a time step of *dt* seconds.  Shape: (n_state, n_state)."""

    @abstractmethod
    def process_noise(self, dt: float) -> np.ndarray:
        """Return Q for a time step of *dt* seconds.  Shape: (n_state, n_state)."""

    def initial_state(
        self, first_obs: np.ndarray, obs_to_state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sensible default initial state and covariance from the first observation.

        Parameters
        ----------
        first_obs:
            The first measurement values (e.g. [x, y]).
        obs_to_state:
            A (n_state,) array that maps observation dimensions to state
            dimensions.  If ``None``, zeros are placed in unmeasured slots
            (velocity = 0 initially).

        Returns
        -------
        x0 : np.ndarray, shape (n_state,)
        P0 : np.ndarray, shape (n_state, n_state)
        """
        x0 = np.zeros(self.n_state)
        x0[: len(first_obs)] = first_obs
        P0 = np.eye(self.n_state) * 1e4  # large initial uncertainty
        return x0, P0


@dataclass
class ConstantVelocity2D(StateSpaceModel):
    """2-D constant-velocity model.

    State vector: [x, y, vx, vy]

    Parameters
    ----------
    process_noise_std:
        Standard deviation of the process noise acceleration (m/s²).
        Higher values allow the filter to track faster-changing velocities
        at the cost of less smoothing.
    """

    process_noise_std: float = 1.0

    @property
    def n_state(self) -> int:
        return 4

    def transition_matrix(self, dt: float) -> np.ndarray:
        """F = I_2 ⊗ [[1, dt], [0, 1]] (block-diagonal)."""
        return np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )

    def process_noise(self, dt: float) -> np.ndarray:
        """Discrete white-noise acceleration model (Singer model).

        Q = σ² * [[dt⁴/4, 0, dt³/2, 0],
                   [0, dt⁴/4, 0, dt³/2],
                   [dt³/2, 0, dt², 0],
                   [0, dt³/2, 0, dt²]]
        """
        q = self.process_noise_std ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        return q * np.array(
            [[dt4 / 4, 0.0, dt3 / 2, 0.0],
             [0.0, dt4 / 4, 0.0, dt3 / 2],
             [dt3 / 2, 0.0, dt2, 0.0],
             [0.0, dt3 / 2, 0.0, dt2]],
            dtype=float,
        )
