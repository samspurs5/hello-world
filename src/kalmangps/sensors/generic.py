"""Generic sensor config and measurement.

``GenericSensorConfig`` is the single parameterisable config class that all
factory-created sensors use.  It binds together:

- An ``ObservationModel`` (the *math*)
- A ``value_extractor``   (how to read measurement values from a DataFrame row)
- A ``covariance_extractor`` (how to read or compute R from a row)

This separation means the same model can be used with different data schemas
without touching the filter math, and new sensor archetypes can be added to
the factory without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar

import numpy as np
import pandas as pd

from .base import Measurement, SensorConfig
from .models import ObservationModel

# Cache of dynamically created measurement classes, keyed by sensor_tag.
_MEAS_CLASS_CACHE: dict[str, type] = {}


def _measurement_class(sensor_tag: str) -> type:
    """Return (or create) a ``GenericMeasurement`` subclass with the given tag."""
    if sensor_tag not in _MEAS_CLASS_CACHE:
        _MEAS_CLASS_CACHE[sensor_tag] = type(
            f"Measurement_{sensor_tag}",
            (GenericMeasurement,),
            {"sensor_type": sensor_tag},
        )
    return _MEAS_CLASS_CACHE[sensor_tag]


@dataclass
class GenericMeasurement(Measurement):
    """Measurement produced by a ``GenericSensorConfig``.

    The ``sensor_type`` class variable is set dynamically per sensor tag so
    that result DataFrames carry a meaningful label (e.g. ``"entrance_cam"``,
    ``"tower_A"``) rather than a generic ``"generic"``.
    """

    sensor_type: ClassVar[str] = "generic"

    @property
    def n_obs(self) -> int:
        return len(self.values)


@dataclass
class GenericSensorConfig(SensorConfig):
    """One parameterisable sensor config backed by an ``ObservationModel``.

    Parameters
    ----------
    model:
        The observation model (provides ``observe``, ``jacobian``,
        ``set_reference``, ``default_values``, ``default_covariance``).
    sensor_tag:
        Short string used as ``sensor_type`` in result outputs.  Defaults to
        the sensor ``name``.
    timestamp_col:
        DataFrame column name for UTC timestamps.
    value_extractor:
        ``(row: pd.Series) -> np.ndarray`` — reads the raw measurement vector
        from a row.  When ``None`` the model's ``default_values()`` is used
        (suitable for fixed-point sensors where the observation is always the
        sensor's own location).
    covariance_extractor:
        ``(row: pd.Series) -> np.ndarray`` — builds the R matrix from a row.
        When ``None`` the model's ``default_covariance(row)`` is used (may
        inspect confidence/quality columns if the model supports it).
    """

    model: ObservationModel = field(default_factory=lambda: (_ for _ in ()).throw(
        TypeError("model is required")))
    sensor_tag: str = "generic"
    timestamp_col: str = "timestamp"
    value_extractor: Callable[[pd.Series], np.ndarray] | None = None
    covariance_extractor: Callable[[pd.Series], np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # SensorConfig interface
    # ------------------------------------------------------------------ #

    @property
    def observation_matrix(self) -> np.ndarray | None:
        """Constant H for linear models; ``None`` for nonlinear ones."""
        # FixedPointModel and CallableModel(observation_matrix_val=...) are linear
        try:
            H = self.model.jacobian(np.zeros(4))
            # Check it's constant (same for a different state)
            H2 = self.model.jacobian(np.ones(4))
            if np.allclose(H, H2):
                return H
        except Exception:
            pass
        return None

    def observe(self, state: np.ndarray) -> np.ndarray:
        return self.model.observe(state)

    def observation_jacobian(self, state: np.ndarray) -> np.ndarray:
        return self.model.jacobian(state)

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        self.model.set_reference(lat_ref, lon_ref)

    def measurement_from_row(self, row: pd.Series) -> GenericMeasurement:
        ts = pd.Timestamp(row[self.timestamp_col])

        values = (
            self.value_extractor(row)
            if self.value_extractor is not None
            else self.model.default_values()
        )
        cov = (
            self.covariance_extractor(row)
            if self.covariance_extractor is not None
            else self.model.default_covariance(row)
        )

        cls = _measurement_class(self.sensor_tag)
        return cls(timestamp=ts, values=np.asarray(values, dtype=float), covariance=cov)
