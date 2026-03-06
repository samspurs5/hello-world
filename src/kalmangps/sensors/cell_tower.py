"""Cell-tower sensor.

Cell towers can contribute location evidence in several ways depending on what
the network reports.  This module implements the three most common measurement
types:

``"range"``
    Distance from the tower estimated from RSSI via a log-distance path-loss
    model.  This is a **nonlinear** scalar measurement:

        h(x) = √((x - tₓ)² + (y - tᵧ)²)

    with Jacobian:

        H(x) = [(x - tₓ)/d,  (y - tᵧ)/d,  0,  0]

    The EKF linearises this at the current state estimate, making it compatible
    with the rest of the fusion pipeline.

``"angle"`` (Angle-of-Arrival)
    Bearing to the tower from the entity's position.  Also nonlinear:

        h(x) = atan2(tᵧ - y,  tₓ - x)

    Jacobian:

        H(x) = [(tᵧ - y)/d²,  -(tₓ - x)/d²,  0,  0]

``"position"``
    The network provides a coarse 2-D position fix (e.g. Cell-ID centroid or
    network-level OTDOA).  This is a linear position observation — same H as
    GPS but with a much larger covariance.

RSSI → range conversion
-----------------------
Uses the log-distance path-loss model::

    RSSI(d) = RSSI_at_1m - 10 · n · log₁₀(d)

Inverting::

    d = 10^((RSSI_at_1m - RSSI) / (10 · n))

Typical values: ``rssi_at_1m ≈ -40 dBm``, ``path_loss_exponent ≈ 2.0–4.0``
(2 = free space, 3.5 = typical urban).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from .base import Measurement, SensorConfig
from .gps import latlon_to_xy

MeasurementType = Literal["range", "angle", "position"]


@dataclass
class CellTowerMeasurement(Measurement):
    """One measurement from a cell tower.

    ``values``     depends on ``measurement_type``:
      - ``"range"``    : [range_m]             — shape (1,)
      - ``"angle"``    : [bearing_rad]         — shape (1,)
      - ``"position"`` : [x_m, y_m]           — shape (2,)
    ``covariance`` : (1,1) or (2,2) accordingly.
    """

    sensor_type: ClassVar[str] = "cell_tower"

    tower_lat: float = 0.0
    tower_lon: float = 0.0
    measurement_type: MeasurementType = "range"
    raw_rssi_dbm: float | None = None

    @property
    def n_obs(self) -> int:
        return 1 if self.measurement_type in ("range", "angle") else 2


@dataclass
class CellTowerConfig(SensorConfig):
    """Sensor configuration for one cell tower.

    For deployments with multiple towers, create one ``CellTowerConfig`` per
    tower and register each as a separate sensor (e.g. ``"tower_A"``,
    ``"tower_B"``).

    Parameters
    ----------
    tower_lat, tower_lon:
        Geographic coordinates of the cell tower.
    measurement_type:
        ``"range"`` (RSSI-derived distance), ``"angle"`` (AoA bearing), or
        ``"position"`` (coarse network fix).
    range_std_m:
        1-sigma range uncertainty in metres.  Typical values: 200–500 m for
        urban macro cells, 50–100 m for small cells.
    angle_std_rad:
        1-sigma bearing uncertainty in radians (``"angle"`` type only).
        Typical: 0.1–0.3 rad.
    position_std_m:
        1-sigma position uncertainty for ``"position"`` type (metres).
    rssi_col:
        DataFrame column name for RSSI in dBm (used to derive range).
    range_col:
        DataFrame column name for a pre-computed range in metres (overrides
        RSSI if present).
    angle_col:
        DataFrame column name for a pre-computed bearing in radians.
    position_lat_col, position_lon_col:
        Column names for ``"position"`` type network fixes.
    rssi_at_1m:
        RSSI (dBm) measured 1 m from the antenna.  Calibration parameter for
        the path-loss model.
    path_loss_exponent:
        Path-loss exponent *n* for the log-distance model.
    min_range_m:
        Minimum plausible range (guards against division-by-zero in the
        Jacobian when the entity is very close to the tower).
    timestamp_col:
        Column name for UTC timestamps.
    """

    name: str = "cell_tower"
    tower_lat: float = 0.0
    tower_lon: float = 0.0
    measurement_type: MeasurementType = "range"

    # Noise parameters
    range_std_m: float = 300.0
    angle_std_rad: float = 0.2
    position_std_m: float = 500.0

    # Column mappings
    rssi_col: str = "rssi_dbm"
    range_col: str | None = None
    angle_col: str | None = None
    position_lat_col: str = "cell_lat"
    position_lon_col: str = "cell_lon"
    timestamp_col: str = "timestamp"

    # Path-loss model (RSSI → range)
    rssi_at_1m: float = -40.0
    path_loss_exponent: float = 3.5
    min_range_m: float = 10.0

    # Set by set_reference()
    _tower_x_m: float = field(default=0.0, init=False, repr=False)
    _tower_y_m: float = field(default=0.0, init=False, repr=False)
    _lat_ref: float = field(default=0.0, init=False, repr=False)
    _lon_ref: float = field(default=0.0, init=False, repr=False)
    _reference_set: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # SensorConfig interface — nonlinear; no static observation_matrix
    # ------------------------------------------------------------------ #

    @property
    def observation_matrix(self) -> None:  # type: ignore[override]
        return None  # nonlinear sensors return None

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        """Compute tower position in local Cartesian frame."""
        tx, ty = latlon_to_xy(self.tower_lat, self.tower_lon, lat_ref, lon_ref)
        self._tower_x_m = float(tx)
        self._tower_y_m = float(ty)
        self._lat_ref = lat_ref
        self._lon_ref = lon_ref
        self._reference_set = True

    def observe(self, state: np.ndarray) -> np.ndarray:
        """Nonlinear observation function h(state)."""
        self._check_reference()
        dx = state[0] - self._tower_x_m
        dy = state[1] - self._tower_y_m

        if self.measurement_type == "range":
            d = max(np.sqrt(dx**2 + dy**2), self.min_range_m)
            return np.array([d])
        elif self.measurement_type == "angle":
            # Bearing from entity to tower
            return np.array([np.arctan2(self._tower_y_m - state[1],
                                        self._tower_x_m - state[0])])
        else:  # "position" — linear fallback
            return np.array([state[0], state[1]])

    def observation_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian ∂h/∂x evaluated at *state*."""
        self._check_reference()
        dx = state[0] - self._tower_x_m
        dy = state[1] - self._tower_y_m

        if self.measurement_type == "range":
            d = max(np.sqrt(dx**2 + dy**2), self.min_range_m)
            return np.array([[dx / d, dy / d, 0.0, 0.0]])

        elif self.measurement_type == "angle":
            d2 = max(dx**2 + dy**2, self.min_range_m**2)
            # h = atan2(ty - y, tx - x)
            # ∂h/∂x[0] = (ty - y) / d²,  ∂h/∂x[1] = -(tx - x) / d²
            return np.array([[(self._tower_y_m - state[1]) / d2,
                               -(self._tower_x_m - state[0]) / d2,
                               0.0, 0.0]])

        else:  # "position"
            return np.array(
                [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0]],
                dtype=float,
            )

    def measurement_from_row(self, row: pd.Series) -> CellTowerMeasurement:
        self._check_reference()
        ts = pd.Timestamp(row[self.timestamp_col])
        raw_rssi = None

        if self.measurement_type == "range":
            value, cov = self._range_from_row(row)
            raw_rssi = float(row[self.rssi_col]) if self.rssi_col in row.index else None

        elif self.measurement_type == "angle":
            value, cov = self._angle_from_row(row)

        else:  # "position"
            value, cov = self._position_from_row(row)

        return CellTowerMeasurement(
            timestamp=ts,
            values=value,
            covariance=cov,
            tower_lat=self.tower_lat,
            tower_lon=self.tower_lon,
            measurement_type=self.measurement_type,
            raw_rssi_dbm=raw_rssi,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_reference(self) -> None:
        if not self._reference_set:
            raise RuntimeError(
                "CellTowerConfig.set_reference() must be called before use. "
                "The fusion pipeline does this automatically."
            )

    def _range_from_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        # Prefer explicit range column; fall back to RSSI conversion
        if self.range_col and self.range_col in row.index:
            val = row[self.range_col]
            if pd.notna(val) and float(val) > 0:
                range_m = float(val)
                return np.array([range_m]), np.array([[self.range_std_m**2]])

        if self.rssi_col in row.index:
            rssi = float(row[self.rssi_col])
            range_m = self._rssi_to_range(rssi)
        else:
            # No data — use a very uncertain range measurement at max plausible distance
            range_m = 1000.0

        return np.array([range_m]), np.array([[self.range_std_m**2]])

    def _angle_from_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        if self.angle_col and self.angle_col in row.index:
            angle_rad = float(row[self.angle_col])
        else:
            raise ValueError(
                f"CellTowerConfig(measurement_type='angle'): "
                f"angle_col='{self.angle_col}' not found in row."
            )
        return np.array([angle_rad]), np.array([[self.angle_std_rad**2]])

    def _position_from_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        lat = float(row[self.position_lat_col])
        lon = float(row[self.position_lon_col])
        x, y = latlon_to_xy(lat, lon, self._lat_ref, self._lon_ref)
        cov = np.diag([self.position_std_m**2, self.position_std_m**2])
        return np.array([float(x), float(y)]), cov

    def _rssi_to_range(self, rssi_dbm: float) -> float:
        """Log-distance path-loss model: d = 10^((A - RSSI) / (10n))."""
        exponent = (self.rssi_at_1m - rssi_dbm) / (10.0 * self.path_loss_exponent)
        return max(10.0 ** exponent, self.min_range_m)
