"""Observation model strategies.

An ``ObservationModel`` encapsulates the **pure mathematics** of one sensor
type — the observation function h(x) and its Jacobian ∂h/∂x — completely
decoupled from how data is read from a DataFrame.

Adding a new sensor type only requires:

1. Subclassing ``ObservationModel`` (or using ``CallableModel``).
2. Registering a factory method with ``SensorFactory.register()``.

No changes to the filter, fusion engine, or pipeline are needed.

Built-in models
---------------
``FixedPointModel``
    Entity detected at a **known location**.  Examples: CCTV / camera trap,
    RFID gate, ANPR reader, Bluetooth beacon, motion sensor, speed camera,
    toll-booth ping, sonar buoy.  The measurement is always the sensor's
    known Cartesian position; uncertainty comes from the detection zone
    radius (optionally scaled by a per-detection confidence score).

``RangeModel``
    Distance from a **fixed infrastructure node**.  Examples: cell-tower RSSI,
    Wi-Fi access point ranging, BLE beacon RSSI, acoustic ranging, radar cross-
    range.  Nonlinear: h(x) = ‖position − node‖.

``BearingModel``
    Angle-of-arrival from a **fixed infrastructure node**.  Examples:
    directional antenna, AoA Bluetooth, phased-array radar azimuth.
    Nonlinear: h(x) = atan2(node_y − y,  node_x − x).

``CallableModel``
    Fully user-defined via ``observe_fn`` and ``jacobian_fn`` callables.
    Use this for any sensor that doesn't fit the above archetypes (barometer,
    IMU, GNSS-RTK differential, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .gps import latlon_to_xy


class ObservationModel(ABC):
    """Abstract strategy for a sensor's mathematical observation model.

    The filter calls only ``observe`` and ``jacobian``.  The factory wraps an
    ``ObservationModel`` inside a ``GenericSensorConfig`` and supplies the
    data-ingestion logic (value/covariance extractors) separately, so the
    model itself stays pure and testable.
    """

    @property
    @abstractmethod
    def n_obs(self) -> int:
        """Dimension of the observation vector."""

    @abstractmethod
    def observe(self, state: np.ndarray) -> np.ndarray:
        """Predicted measurement h(state), shape ``(n_obs,)``."""

    @abstractmethod
    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian ∂h/∂x at *state*, shape ``(n_obs, n_state)``."""

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        """Recompute internal Cartesian coordinates when the reference changes.

        Called once per segment by the fusion pipeline before any measurements
        are processed.  Models with a fixed geographic location (cell tower,
        camera, etc.) must override this.
        """

    def default_values(self) -> np.ndarray:
        """Measurement values when the row carries no raw data.

        Only ``FixedPointModel`` implements this — the observation is always
        the sensor's known Cartesian position regardless of row content.
        Other models must be paired with a ``value_extractor`` in the factory.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no default_values(); "
            "supply a value_extractor in the factory call."
        )

    def default_covariance(self, row: pd.Series) -> np.ndarray:
        """Default R matrix for a given row.

        May inspect row columns (e.g. a confidence score) to scale uncertainty
        dynamically.  Called when no ``covariance_extractor`` is provided.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no default_covariance(); "
            "supply a covariance_extractor in the factory call."
        )


# --------------------------------------------------------------------------- #
# Fixed-point (linear)
# --------------------------------------------------------------------------- #

@dataclass
class FixedPointModel(ObservationModel):
    """Detection at a fixed geographic location.

    Covers any sensor that detects an entity *at a known place*: cameras,
    RFID gates, ANPR readers, Bluetooth beacons, motion sensors, etc.

    Observation model:  h(x) = H · x  where H = [[1,0,0,0],[0,1,0,0]]
    Measurement values: z = [node_x_m, node_y_m]  (the sensor's known location)
    Innovation:         y = z − h(x) = [node_x − state_x, node_y − state_y]

    Parameters
    ----------
    lat, lon:
        Geographic coordinates of the sensor.
    uncertainty_m:
        Base 1-sigma uncertainty radius (metres).  Think of this as the
        physical detection zone.
    confidence_col:
        Optional DataFrame column with a confidence/score in [0, 1].  When
        present the effective radius is ``uncertainty_m / confidence``, so a
        high-confidence detection tightens the Kalman update.
    min_confidence:
        Floor on the confidence value (prevents division by zero).
    """

    lat: float
    lon: float
    uncertainty_m: float = 50.0
    confidence_col: str | None = None
    min_confidence: float = 0.05

    _x_m: float = field(default=0.0, init=False, repr=False)
    _y_m: float = field(default=0.0, init=False, repr=False)
    _reference_set: bool = field(default=False, init=False, repr=False)

    # Constant H — linear sensor
    _H: np.ndarray = field(
        default_factory=lambda: np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
        init=False,
        repr=False,
    )

    @property
    def n_obs(self) -> int:
        return 2

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        x, y = latlon_to_xy(self.lat, self.lon, lat_ref, lon_ref)
        self._x_m = float(x)
        self._y_m = float(y)
        self._reference_set = True

    def observe(self, state: np.ndarray) -> np.ndarray:
        """Predicted entity position H·x."""
        return self._H @ state

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        return self._H

    def default_values(self) -> np.ndarray:
        """The sensor's known Cartesian position (the actual observation z)."""
        self._require_reference()
        return np.array([self._x_m, self._y_m])

    def default_covariance(self, row: pd.Series) -> np.ndarray:
        """Confidence-scaled diagonal covariance."""
        confidence = 1.0
        if self.confidence_col and self.confidence_col in row.index:
            val = row[self.confidence_col]
            if pd.notna(val):
                confidence = float(np.clip(val, self.min_confidence, 1.0))
        r = self.uncertainty_m / max(confidence, self.min_confidence)
        return np.diag([r**2, r**2])

    def _require_reference(self) -> None:
        if not self._reference_set:
            raise RuntimeError(
                f"FixedPointModel at ({self.lat}, {self.lon}) needs "
                "set_reference() before use.  The fusion pipeline calls this "
                "automatically; call it manually for standalone use."
            )


# --------------------------------------------------------------------------- #
# Range (nonlinear)
# --------------------------------------------------------------------------- #

@dataclass
class RangeModel(ObservationModel):
    """Distance from a fixed infrastructure node (nonlinear).

    Covers: cell-tower RSSI, Wi-Fi/BLE access-point ranging, sonar,
    acoustic ping, radar slant range, etc.

    Observation model:  h(x) = √((x − node_x)² + (y − node_y)²)
    Jacobian:           H(x) = [(x − nx)/d,  (y − ny)/d,  0,  0]

    Parameters
    ----------
    lat, lon:
        Geographic coordinates of the node.
    range_std_m:
        1-sigma range uncertainty in metres.
    min_range_m:
        Minimum clamp on the denominator (prevents divide-by-zero when the
        entity is very close to the node).
    rssi_at_1m:
        Received signal strength at 1 m (dBm).  Used for RSSI → range.
    path_loss_exponent:
        Log-distance path-loss exponent *n*.  Typical: 2.0 (free space) –
        4.0 (heavy urban / indoors).
    """

    lat: float
    lon: float
    range_std_m: float = 300.0
    min_range_m: float = 10.0
    rssi_at_1m: float = -40.0
    path_loss_exponent: float = 3.5

    _x_m: float = field(default=0.0, init=False, repr=False)
    _y_m: float = field(default=0.0, init=False, repr=False)
    _reference_set: bool = field(default=False, init=False, repr=False)

    @property
    def n_obs(self) -> int:
        return 1

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        x, y = latlon_to_xy(self.lat, self.lon, lat_ref, lon_ref)
        self._x_m = float(x)
        self._y_m = float(y)
        self._reference_set = True

    def observe(self, state: np.ndarray) -> np.ndarray:
        self._require_reference()
        dx = state[0] - self._x_m
        dy = state[1] - self._y_m
        return np.array([max(np.sqrt(dx**2 + dy**2), self.min_range_m)])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        self._require_reference()
        dx = state[0] - self._x_m
        dy = state[1] - self._y_m
        d = max(np.sqrt(dx**2 + dy**2), self.min_range_m)
        return np.array([[dx / d, dy / d, 0.0, 0.0]])

    def default_covariance(self, row: pd.Series) -> np.ndarray:
        return np.array([[self.range_std_m**2]])

    def rssi_to_range(self, rssi_dbm: float) -> float:
        """Log-distance path-loss inversion: d = 10^((A − RSSI) / (10n))."""
        exp = (self.rssi_at_1m - rssi_dbm) / (10.0 * self.path_loss_exponent)
        return max(10.0**exp, self.min_range_m)

    def _require_reference(self) -> None:
        if not self._reference_set:
            raise RuntimeError(
                f"RangeModel at ({self.lat}, {self.lon}) needs set_reference()."
            )


# --------------------------------------------------------------------------- #
# Bearing (nonlinear)
# --------------------------------------------------------------------------- #

@dataclass
class BearingModel(ObservationModel):
    """Angle-of-arrival from a fixed infrastructure node (nonlinear).

    Covers: directional antenna, AoA Bluetooth, phased-array azimuth.

    Observation model:  h(x) = atan2(node_y − y,  node_x − x)
    Jacobian:
        H(x) = [(node_y − y)/d²,  −(node_x − x)/d²,  0,  0]

    Parameters
    ----------
    lat, lon:
        Geographic coordinates of the receiving node.
    angle_std_rad:
        1-sigma bearing uncertainty (radians).
    min_range_m:
        Minimum distance clamp for the Jacobian denominator.
    """

    lat: float
    lon: float
    angle_std_rad: float = 0.2
    min_range_m: float = 10.0

    _x_m: float = field(default=0.0, init=False, repr=False)
    _y_m: float = field(default=0.0, init=False, repr=False)
    _reference_set: bool = field(default=False, init=False, repr=False)

    @property
    def n_obs(self) -> int:
        return 1

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        x, y = latlon_to_xy(self.lat, self.lon, lat_ref, lon_ref)
        self._x_m = float(x)
        self._y_m = float(y)
        self._reference_set = True

    def observe(self, state: np.ndarray) -> np.ndarray:
        self._require_reference()
        return np.array([np.arctan2(self._y_m - state[1], self._x_m - state[0])])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        self._require_reference()
        dx = state[0] - self._x_m
        dy = state[1] - self._y_m
        d2 = max(dx**2 + dy**2, self.min_range_m**2)
        return np.array([[(self._y_m - state[1]) / d2,
                          -(self._x_m - state[0]) / d2,
                          0.0, 0.0]])

    def default_covariance(self, row: pd.Series) -> np.ndarray:
        return np.array([[self.angle_std_rad**2]])

    def _require_reference(self) -> None:
        if not self._reference_set:
            raise RuntimeError(
                f"BearingModel at ({self.lat}, {self.lon}) needs set_reference()."
            )


# --------------------------------------------------------------------------- #
# Callable (user-defined)
# --------------------------------------------------------------------------- #

@dataclass
class CallableModel(ObservationModel):
    """Fully user-defined observation model via callables.

    Use this for sensors that don't fit the built-in archetypes: barometers,
    IMUs, GNSS-RTK differentials, custom signal-strength models, etc.

    Parameters
    ----------
    observe_fn:
        ``h(state: np.ndarray) -> np.ndarray`` — the observation function.
    jacobian_fn:
        ``H(state: np.ndarray) -> np.ndarray`` — Jacobian ∂h/∂x.
        If ``None``, ``observation_matrix_val`` is used as a constant H.
    _n_obs:
        Dimension of the observation vector.
    default_cov:
        Default R matrix, shape ``(n_obs, n_obs)``.
    observation_matrix_val:
        Constant H for linear sensors (alternative to ``jacobian_fn``).
    reference_fn:
        Optional ``(lat_ref, lon_ref) -> None`` hook, useful when
        ``observe_fn`` / ``jacobian_fn`` close over mutable geometry.
    """

    observe_fn: Callable[[np.ndarray], np.ndarray] | None = None
    jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None
    _n_obs: int = 1
    default_cov: np.ndarray = field(default_factory=lambda: np.eye(1))
    observation_matrix_val: np.ndarray | None = None
    reference_fn: Callable[[float, float], None] | None = None

    def __post_init__(self) -> None:
        if self.observe_fn is None and self.observation_matrix_val is None:
            raise ValueError(
                "CallableModel requires either observe_fn (nonlinear) or "
                "observation_matrix_val (linear)."
            )

    @property
    def n_obs(self) -> int:
        if self.observation_matrix_val is not None:
            return self.observation_matrix_val.shape[0]
        return self._n_obs

    def observe(self, state: np.ndarray) -> np.ndarray:
        if self.observe_fn is not None:
            return self.observe_fn(state)
        assert self.observation_matrix_val is not None
        return self.observation_matrix_val @ state

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        if self.jacobian_fn is not None:
            return self.jacobian_fn(state)
        if self.observation_matrix_val is not None:
            return self.observation_matrix_val
        raise NotImplementedError("No jacobian_fn or observation_matrix_val.")

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        if self.reference_fn is not None:
            self.reference_fn(lat_ref, lon_ref)

    def default_covariance(self, row: pd.Series) -> np.ndarray:
        return self.default_cov.copy()
