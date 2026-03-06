"""Custom / plug-in sensor for arbitrary observation models.

When none of the built-in sensors fit your use case you can define a sensor
entirely through callable hooks, without subclassing:

.. code-block:: python

    from kalmangps.sensors import CustomSensorConfig

    def my_observe(state):
        # e.g. altitude from barometer: observe state[1] (y == north, but
        # your model may use state[2] as height in a 3-D extension)
        return np.array([state[2]])

    def my_jacobian(state):
        return np.array([[0., 0., 1., 0.]])

    baro = CustomSensorConfig(
        name="barometer",
        sensor_type_tag="baro",
        observe_fn=my_observe,
        jacobian_fn=my_jacobian,
        default_covariance=np.array([[4.0]]),   # 2 m 1-sigma
        timestamp_col="time",
        value_cols=["altitude_m"],
    )

Column-based ingestion
----------------------
``value_cols`` lists the DataFrame column(s) that provide the raw measurement
values.  ``covariance_cols`` optionally lists columns for per-row variance
values (diagonal only; one entry per value column).  If absent,
``default_covariance`` is used for every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np
import pandas as pd

from .base import Measurement, SensorConfig


@dataclass
class CustomMeasurement(Measurement):
    """Measurement produced by a :class:`CustomSensorConfig`."""

    #: Populated at class creation time from ``sensor_type_tag``.
    sensor_type: ClassVar[str] = "custom"

    @property
    def n_obs(self) -> int:
        return len(self.values)


def _make_custom_measurement_class(tag: str) -> type[CustomMeasurement]:
    """Dynamically create a CustomMeasurement subclass with a fixed sensor_type."""
    return type(
        f"CustomMeasurement_{tag}",
        (CustomMeasurement,),
        {"sensor_type": tag},
    )


@dataclass
class CustomSensorConfig(SensorConfig):
    """Plug-in sensor defined entirely by callables.

    Parameters
    ----------
    name:
        Human-readable label.
    sensor_type_tag:
        Short string identifier used in results (e.g. ``"baro"``, ``"wifi"``).
    observe_fn:
        ``h(state) -> np.ndarray`` — the observation function.
    jacobian_fn:
        ``H(state) -> np.ndarray`` — Jacobian ∂h/∂x.
        Pass ``None`` to use a constant ``observation_matrix`` instead.
    observation_matrix_val:
        Constant H matrix for linear sensors.  Used when ``jacobian_fn`` is
        ``None``.  Exactly one of ``jacobian_fn`` / ``observation_matrix_val``
        must be provided.
    default_covariance:
        Default R matrix (shape ``(n_obs, n_obs)``).  Used when rows do not
        carry their own covariance.
    timestamp_col:
        Column name for UTC timestamps.
    value_cols:
        Ordered list of column names whose values form the measurement vector.
    covariance_cols:
        Optional list of column names for per-row diagonal variances.  Must
        have the same length as ``value_cols``.
    reference_fn:
        Optional ``(lat_ref, lon_ref) -> None`` hook called by
        ``set_reference()``.  Useful when the sensor's observe/jacobian
        functions close over mutable state (e.g. a tower location).
    """

    name: str = "custom"
    sensor_type_tag: str = "custom"
    observe_fn: Callable[[np.ndarray], np.ndarray] | None = None
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None
    observation_matrix_val: np.ndarray | None = None
    default_covariance: np.ndarray = field(default_factory=lambda: np.eye(1))
    timestamp_col: str = "timestamp"
    value_cols: list[str] = field(default_factory=list)
    covariance_cols: list[str] | None = None
    reference_fn: Callable[[float, float], None] | None = None

    def __post_init__(self) -> None:
        if self.observe_fn is None and self.observation_matrix_val is None:
            raise ValueError(
                "Provide either observe_fn (nonlinear) or "
                "observation_matrix_val (linear)."
            )
        self._meas_class = _make_custom_measurement_class(self.sensor_type_tag)

    # ------------------------------------------------------------------ #
    # SensorConfig interface
    # ------------------------------------------------------------------ #

    @property
    def observation_matrix(self) -> np.ndarray | None:
        return self.observation_matrix_val

    def observe(self, state: np.ndarray) -> np.ndarray:
        if self.observe_fn is not None:
            return self.observe_fn(state)
        # linear fallback
        assert self.observation_matrix_val is not None
        return self.observation_matrix_val @ state

    def observation_jacobian(self, state: np.ndarray) -> np.ndarray:
        if self.jacobian_fn is not None:
            return self.jacobian_fn(state)
        if self.observation_matrix_val is not None:
            return self.observation_matrix_val
        raise NotImplementedError("No jacobian_fn or observation_matrix_val provided.")

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        if self.reference_fn is not None:
            self.reference_fn(lat_ref, lon_ref)

    def measurement_from_row(self, row: pd.Series) -> CustomMeasurement:
        if not self.value_cols:
            raise ValueError("CustomSensorConfig.value_cols must be non-empty.")

        values = np.array([float(row[c]) for c in self.value_cols])

        if self.covariance_cols:
            variances = np.array([float(row[c]) for c in self.covariance_cols])
            cov = np.diag(variances)
        else:
            cov = self.default_covariance.copy()

        return self._meas_class(
            timestamp=pd.Timestamp(row[self.timestamp_col]),
            values=values,
            covariance=cov,
        )
