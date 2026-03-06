"""GPS sensor implementation.

Converts raw GPS observations (latitude, longitude, optional accuracy) into
``GPSMeasurement`` objects expressed in a **local East-North Cartesian frame**
centred on a chosen reference point.

Working in metres (not degrees) means the state-space model's units are
physically meaningful and avoids the non-uniform scaling between lat and lon
degrees at different latitudes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import Measurement, SensorConfig

# WGS-84 mean Earth radius in metres, used for small-area projections.
_METRES_PER_DEGREE_LAT: float = 111_139.0


def _metres_per_degree_lon(lat_deg: float) -> float:
    """Longitude scale at a given latitude (metres per degree)."""
    return _METRES_PER_DEGREE_LAT * np.cos(np.radians(lat_deg))


def latlon_to_xy(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    lat_ref: float,
    lon_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lat, lon) to local (x=East, y=North) in metres."""
    x = (np.asarray(lon) - lon_ref) * _metres_per_degree_lon(lat_ref)
    y = (np.asarray(lat) - lat_ref) * _METRES_PER_DEGREE_LAT
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def xy_to_latlon(
    x: float | np.ndarray,
    y: float | np.ndarray,
    lat_ref: float,
    lon_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the local projection back to (lat, lon)."""
    lat = np.asarray(y) / _METRES_PER_DEGREE_LAT + lat_ref
    lon = np.asarray(x) / _metres_per_degree_lon(lat_ref) + lon_ref
    return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)


@dataclass
class GPSMeasurement(Measurement):
    """One GPS fix in a local Cartesian frame.

    ``values`` = [x_east_m, y_north_m]
    ``covariance`` = 2×2 diagonal built from ``accuracy_m``
      (horizontal 1-sigma error in metres, as reported by the GPS chipset).

    When ``accuracy_m`` is unknown / not reported the caller should supply a
    default, e.g. ``GPSConfig.default_accuracy_m``.
    """

    sensor_type: ClassVar[str] = "gps"

    #: Raw geographic coordinates (preserved for output / debugging).
    lat: float = 0.0
    lon: float = 0.0
    #: Reported horizontal 1-sigma accuracy (metres).
    accuracy_m: float = 10.0

    @property
    def n_obs(self) -> int:
        return 2


@dataclass
class GPSConfig(SensorConfig):
    """Sensor configuration for a GPS receiver.

    Parameters
    ----------
    lat_col, lon_col:
        Column names for latitude and longitude in the input DataFrame.
    accuracy_col:
        Optional column name that holds the GPS horizontal accuracy in metres.
        If the column is absent or a row's value is NaN, ``default_accuracy_m``
        is used.
    timestamp_col:
        Column name for the UTC timestamp.
    default_accuracy_m:
        Fallback accuracy when none is reported (metres).
    lat_ref, lon_ref:
        Reference point for the local Cartesian projection.  Set automatically
        from the first segment if left as ``None``.
    """

    name: str = "gps"
    lat_col: str = "lat"
    lon_col: str = "lon"
    accuracy_col: str | None = "accuracy"
    timestamp_col: str = "timestamp"
    default_accuracy_m: float = 10.0
    lat_ref: float | None = None
    lon_ref: float | None = None

    # ------------------------------------------------------------------ #
    # SensorConfig interface
    # ------------------------------------------------------------------ #

    @property
    def observation_matrix(self) -> np.ndarray:
        """H maps [x, y, vx, vy] -> [x, y]."""
        return np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=float,
        )

    def measurement_from_row(self, row: pd.Series) -> GPSMeasurement:
        lat = float(row[self.lat_col])
        lon = float(row[self.lon_col])

        # Accuracy
        accuracy = self.default_accuracy_m
        if self.accuracy_col and self.accuracy_col in row.index:
            val = row[self.accuracy_col]
            if pd.notna(val) and float(val) > 0:
                accuracy = float(val)

        # Projection
        lat_ref = self.lat_ref if self.lat_ref is not None else lat
        lon_ref = self.lon_ref if self.lon_ref is not None else lon
        x, y = latlon_to_xy(lat, lon, lat_ref, lon_ref)

        cov = np.diag([accuracy**2, accuracy**2])

        return GPSMeasurement(
            timestamp=pd.Timestamp(row[self.timestamp_col]),
            values=np.array([x, y], dtype=float),
            covariance=cov,
            lat=lat,
            lon=lon,
            accuracy_m=accuracy,
        )

    # ------------------------------------------------------------------ #
    # Batch helpers
    # ------------------------------------------------------------------ #

    def measurements_from_df(
        self, df: pd.DataFrame, *, auto_ref: bool = True
    ) -> list[GPSMeasurement]:
        """Convert an entire DataFrame segment into a list of measurements.

        Parameters
        ----------
        df:
            DataFrame slice for one trajectory segment.
        auto_ref:
            When ``True`` and ``lat_ref``/``lon_ref`` are not yet set, derive
            them from the first row of *this* segment so the Cartesian
            coordinates remain small and numerically stable.
        """
        if auto_ref and (self.lat_ref is None or self.lon_ref is None):
            first = df.iloc[0]
            self.lat_ref = float(first[self.lat_col])
            self.lon_ref = float(first[self.lon_col])

        return [self.measurement_from_row(row) for _, row in df.iterrows()]
