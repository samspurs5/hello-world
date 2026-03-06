"""Capture-recapture sensor.

A capture-recapture event fires when a stationary reader *detects* a passing
entity.  Examples:

- Wildlife camera trap with GPS-tagged animals
- RFID antenna / reader gate (warehouse, livestock, access control)
- Automatic Number Plate Recognition (ANPR / LPR) camera
- Bluetooth / Wi-Fi beacon scanner at a known location
- Toll-booth transponder ping

The measurement model
---------------------
The reader's geographic location is known and fixed.  When a detection occurs
the entity *must have been near the reader*, so the observation is:

    z = [x_reader, y_reader]   (local Cartesian, metres)

with covariance:

    R = diag([σ², σ²])   where σ = detection_radius_m

This is a **linear** position observation (same H as GPS) with potentially
much larger uncertainty — a camera at a road junction might have σ = 50 m
while an RFID gate might have σ = 2 m.

Optional confidence column
--------------------------
When each detection row carries a ``confidence`` value in [0, 1] (e.g. image
recognition score, RSSI strength normalised), the effective radius is scaled
inversely:

    σ_effective = detection_radius_m / max(confidence, min_confidence)

So a high-confidence detection tightens the uncertainty, effectively
telling the filter "the entity was very close to the reader".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from .base import Measurement, SensorConfig
from .gps import latlon_to_xy


@dataclass
class CaptureRecaptureMeasurement(Measurement):
    """Detection event at a fixed reader location.

    ``values``     = [x_reader_m, y_reader_m] in local Cartesian frame.
    ``covariance`` = 2×2 diagonal, scaled by effective detection radius.
    """

    sensor_type: ClassVar[str] = "capture_recapture"

    #: Raw reader geographic coordinates (for diagnostics).
    reader_lat: float = 0.0
    reader_lon: float = 0.0
    #: Effective 1-sigma detection radius used (metres).
    effective_radius_m: float = 50.0
    #: Detection confidence in [0, 1], if available.
    confidence: float = 1.0

    @property
    def n_obs(self) -> int:
        return 2


@dataclass
class CaptureRecaptureConfig(SensorConfig):
    """Sensor configuration for a capture-recapture reader at a fixed location.

    Parameters
    ----------
    reader_lat, reader_lon:
        Geographic coordinates of the reader / detector.
    detection_radius_m:
        Base 1-sigma uncertainty (metres) for a detection with full confidence.
        Think of this as the physical detection zone radius.
    confidence_col:
        Optional DataFrame column name carrying a detection confidence in
        ``[0, 1]``.  When present, the effective radius is
        ``detection_radius_m / max(confidence, min_confidence)``.
    min_confidence:
        Floor on the confidence value to prevent division by zero.
    timestamp_col:
        Column name for the UTC timestamp.
    name:
        Human-readable sensor label (useful when multiple readers exist).
    """

    name: str = "capture_recapture"
    reader_lat: float = 0.0
    reader_lon: float = 0.0
    detection_radius_m: float = 50.0
    confidence_col: str | None = None
    min_confidence: float = 0.05
    timestamp_col: str = "timestamp"

    # Set by set_reference(); Cartesian coords of this reader.
    _reader_x_m: float = field(default=0.0, init=False, repr=False)
    _reader_y_m: float = field(default=0.0, init=False, repr=False)
    _reference_set: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # SensorConfig interface
    # ------------------------------------------------------------------ #

    @property
    def observation_matrix(self) -> np.ndarray:
        """H maps [x, y, vx, vy] -> [x_reader, y_reader]."""
        return np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=float,
        )

    def set_reference(self, lat_ref: float, lon_ref: float) -> None:
        """Project the reader's lat/lon into the local Cartesian frame."""
        self._reader_x_m, self._reader_y_m = latlon_to_xy(
            self.reader_lat, self.reader_lon, lat_ref, lon_ref
        )
        self._reader_x_m = float(self._reader_x_m)
        self._reader_y_m = float(self._reader_y_m)
        self._reference_set = True

    def measurement_from_row(self, row: pd.Series) -> CaptureRecaptureMeasurement:
        if not self._reference_set:
            raise RuntimeError(
                "CaptureRecaptureConfig.set_reference() must be called before "
                "measurement_from_row().  The fusion pipeline does this automatically."
            )

        # Confidence-scaled radius
        confidence = 1.0
        if self.confidence_col and self.confidence_col in row.index:
            val = row[self.confidence_col]
            if pd.notna(val):
                confidence = float(np.clip(val, self.min_confidence, 1.0))
        effective_radius = self.detection_radius_m / max(confidence, self.min_confidence)
        cov = np.diag([effective_radius**2, effective_radius**2])

        return CaptureRecaptureMeasurement(
            timestamp=pd.Timestamp(row[self.timestamp_col]),
            values=np.array([self._reader_x_m, self._reader_y_m], dtype=float),
            covariance=cov,
            reader_lat=self.reader_lat,
            reader_lon=self.reader_lon,
            effective_radius_m=effective_radius,
            confidence=confidence,
        )
