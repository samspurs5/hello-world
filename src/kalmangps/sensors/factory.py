"""Sensor factory with a type registry.

``SensorFactory`` is the primary public API for creating secondary sensors.
It ships with factory methods for the common archetypes and exposes a
``register`` decorator so third-party code can extend it without modifying
this file.

Built-in factory methods
------------------------
``SensorFactory.fixed_point(lat, lon, ...)``
    Any sensor that records a *detection at a known location*: cameras,
    RFID gates, ANPR, Bluetooth beacons, motion detectors, speed cameras,
    wildlife traps, etc.

``SensorFactory.range_sensor(lat, lon, ...)``
    Distance from a fixed infrastructure node: cell tower RSSI, Wi-Fi /
    BLE RSSI ranging, sonar ping, radar slant range, etc.

``SensorFactory.bearing_sensor(lat, lon, ...)``
    Angle-of-arrival from a fixed node: directional antenna, AoA Bluetooth,
    phased-array azimuth.

``SensorFactory.callable_sensor(observe_fn, jacobian_fn, ...)``
    Fully user-defined model.  Use for anything that doesn't fit the above:
    barometer, IMU, GNSS-RTK differential, custom fingerprinting, etc.

Registry
--------
Register new sensor types so they can be created by name::

    @SensorFactory.register("wifi_ap")
    def _wifi_ap(lat, lon, ssid=None, **kwargs):
        return SensorFactory.range_sensor(
            lat, lon, rssi_at_1m=-30, path_loss_exponent=2.0, **kwargs
        )

    sensor = SensorFactory.create("wifi_ap", lat=51.5, lon=-0.1, name="ap_foyer")

GPS
---
GPS is handled separately via ``GPSConfig`` because it reads lat/lon from the
DataFrame and sets the projection reference dynamically — it is the *primary*
sensor and has no fixed geographic location.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .generic import GenericSensorConfig
from .models import BearingModel, CallableModel, FixedPointModel, RangeModel


class SensorFactory:
    """Factory for creating ``GenericSensorConfig`` instances.

    Use ``SensorFactory.fixed_point()``, ``SensorFactory.range_sensor()``, etc.
    for common archetypes, or ``SensorFactory.create("type_name", ...)`` to
    instantiate registered custom types.
    """

    _registry: dict[str, Callable[..., GenericSensorConfig]] = {}

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator: register a factory function under *name*.

        .. code-block:: python

            @SensorFactory.register("ble_beacon")
            def _ble_beacon(lat, lon, **kwargs):
                return SensorFactory.range_sensor(
                    lat, lon,
                    rssi_at_1m=-59,
                    path_loss_exponent=2.0,
                    **kwargs,
                )

            sensor = SensorFactory.create("ble_beacon", lat=51.5, lon=-0.1)
        """
        def decorator(fn: Callable) -> Callable:
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def create(cls, sensor_type: str, **kwargs: Any) -> GenericSensorConfig:
        """Instantiate a registered sensor type by name.

        Parameters
        ----------
        sensor_type:
            Name as registered via ``@SensorFactory.register(name)``.
        **kwargs:
            Forwarded verbatim to the registered factory function.

        Raises
        ------
        KeyError
            If *sensor_type* has not been registered.
        """
        if sensor_type not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(
                f"Unknown sensor type '{sensor_type}'. "
                f"Available: {available or '(none registered yet)'}"
            )
        return cls._registry[sensor_type](**kwargs)

    @classmethod
    def registered_types(cls) -> list[str]:
        """Return a sorted list of all registered sensor type names."""
        return sorted(cls._registry)

    # ------------------------------------------------------------------ #
    # Built-in factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def fixed_point(
        cls,
        lat: float,
        lon: float,
        uncertainty_m: float = 50.0,
        *,
        confidence_col: str | None = None,
        min_confidence: float = 0.05,
        name: str = "fixed_point",
        sensor_tag: str | None = None,
        timestamp_col: str = "timestamp",
    ) -> GenericSensorConfig:
        """Sensor that detects an entity at a known geographic location.

        Suitable for **any** fixed-location detection event: CCTV / camera
        trap, RFID gate, ANPR reader, Bluetooth / Wi-Fi beacon, motion
        sensor, speed camera, wildlife trap, toll-booth ping, sonar buoy, …

        Parameters
        ----------
        lat, lon:
            Geographic coordinates of the sensor (decimal degrees).
        uncertainty_m:
            Base 1-σ detection zone radius in metres.  E.g. 2 m for an RFID
            gate, 30 m for a wide-angle camera, 100 m for a motion sensor.
        confidence_col:
            Optional DataFrame column with a per-detection confidence/score
            in [0, 1].  Higher confidence → tighter uncertainty radius.
            Set ``None`` to use ``uncertainty_m`` for every detection.
        min_confidence:
            Floor on the confidence value to prevent division by zero.
        name:
            Human-readable sensor label (used in logging and result metadata).
        sensor_tag:
            Short string used as ``sensor_type`` in result DataFrames.
            Defaults to *name*.
        timestamp_col:
            DataFrame column name for UTC timestamps.

        Examples
        --------
        Camera at a road junction with an image-recognition confidence score::

            cam = SensorFactory.fixed_point(
                lat=51.5074, lon=-0.1278,
                uncertainty_m=20.0,
                confidence_col="detection_score",
                name="junction_camera",
            )

        RFID gate at a known location::

            gate = SensorFactory.fixed_point(
                lat=51.501, lon=-0.089,
                uncertainty_m=2.0,
                name="rfid_gate_1",
            )
        """
        model = FixedPointModel(
            lat=lat, lon=lon,
            uncertainty_m=uncertainty_m,
            confidence_col=confidence_col,
            min_confidence=min_confidence,
        )
        return GenericSensorConfig(
            name=name,
            model=model,
            sensor_tag=sensor_tag or name,
            timestamp_col=timestamp_col,
            value_extractor=None,        # model.default_values() = sensor position
            covariance_extractor=None,   # model.default_covariance(row) = confidence-scaled
        )

    @classmethod
    def range_sensor(
        cls,
        lat: float,
        lon: float,
        range_std_m: float = 300.0,
        *,
        rssi_col: str = "rssi_dbm",
        range_col: str | None = None,
        rssi_at_1m: float = -40.0,
        path_loss_exponent: float = 3.5,
        min_range_m: float = 10.0,
        name: str = "range_sensor",
        sensor_tag: str | None = None,
        timestamp_col: str = "timestamp",
    ) -> GenericSensorConfig:
        """Sensor that measures distance from a fixed infrastructure node.

        Suitable for: cell-tower RSSI, Wi-Fi / BLE RSSI ranging, sonar,
        acoustic ping, radar slant range, ultrasonic sensor, …

        Parameters
        ----------
        lat, lon:
            Geographic coordinates of the infrastructure node.
        range_std_m:
            1-σ range uncertainty in metres.  Typical: 200–500 m for macro
            cell towers, 5–20 m for BLE beacons, 1–5 m for ultrasonic.
        rssi_col:
            DataFrame column name for RSSI in dBm.  Used when *range_col*
            is absent or NaN to compute range via the path-loss model.
        range_col:
            Optional DataFrame column with a pre-computed range in metres.
            Takes priority over the RSSI conversion.
        rssi_at_1m:
            Signal strength at 1 m from the node (dBm).
        path_loss_exponent:
            Log-distance exponent *n*.  2.0 = free space, 3.5 = urban.
        min_range_m:
            Minimum plausible range; clamps the EKF Jacobian denominator.
        name, sensor_tag, timestamp_col:
            See :meth:`fixed_point`.
        """
        model = RangeModel(
            lat=lat, lon=lon,
            range_std_m=range_std_m,
            min_range_m=min_range_m,
            rssi_at_1m=rssi_at_1m,
            path_loss_exponent=path_loss_exponent,
        )

        def _extract_values(row: pd.Series) -> np.ndarray:
            if range_col and range_col in row.index:
                val = row[range_col]
                if pd.notna(val) and float(val) > 0:
                    return np.array([float(val)])
            if rssi_col in row.index and pd.notna(row[rssi_col]):
                return np.array([model.rssi_to_range(float(row[rssi_col]))])
            return np.array([1_000.0])  # unknown range → large fallback

        return GenericSensorConfig(
            name=name,
            model=model,
            sensor_tag=sensor_tag or name,
            timestamp_col=timestamp_col,
            value_extractor=_extract_values,
            covariance_extractor=None,  # model.default_covariance = [[range_std^2]]
        )

    @classmethod
    def bearing_sensor(
        cls,
        lat: float,
        lon: float,
        angle_std_rad: float = 0.2,
        *,
        angle_col: str = "bearing_rad",
        min_range_m: float = 10.0,
        name: str = "bearing_sensor",
        sensor_tag: str | None = None,
        timestamp_col: str = "timestamp",
    ) -> GenericSensorConfig:
        """Sensor that measures angle-of-arrival from a fixed node.

        Suitable for: directional antenna, AoA Bluetooth, phased-array
        radar azimuth, directional microphone, …

        Parameters
        ----------
        lat, lon:
            Geographic coordinates of the receiving node.
        angle_std_rad:
            1-σ bearing uncertainty in radians.  Typical: 0.05–0.3 rad.
        angle_col:
            DataFrame column name for the bearing in radians.
        min_range_m, name, sensor_tag, timestamp_col:
            See :meth:`range_sensor`.
        """
        model = BearingModel(
            lat=lat, lon=lon,
            angle_std_rad=angle_std_rad,
            min_range_m=min_range_m,
        )

        def _extract_values(row: pd.Series) -> np.ndarray:
            if angle_col not in row.index or pd.isna(row[angle_col]):
                raise ValueError(
                    f"bearing_sensor '{name}': column '{angle_col}' missing or NaN."
                )
            return np.array([float(row[angle_col])])

        return GenericSensorConfig(
            name=name,
            model=model,
            sensor_tag=sensor_tag or name,
            timestamp_col=timestamp_col,
            value_extractor=_extract_values,
            covariance_extractor=None,
        )

    @classmethod
    def callable_sensor(
        cls,
        observe_fn: Callable[[np.ndarray], np.ndarray] | None,
        *,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        observation_matrix: np.ndarray | None = None,
        default_covariance: np.ndarray,
        value_extractor: Callable[[pd.Series], np.ndarray],
        covariance_extractor: Callable[[pd.Series], np.ndarray] | None = None,
        n_obs: int = 1,
        reference_fn: Callable[[float, float], None] | None = None,
        name: str = "custom",
        sensor_tag: str | None = None,
        timestamp_col: str = "timestamp",
    ) -> GenericSensorConfig:
        """Fully user-defined sensor via callables.

        Use this for any sensor that doesn't fit the built-in archetypes:
        barometer, IMU, GNSS-RTK differential, custom fingerprinting, …

        Parameters
        ----------
        observe_fn:
            ``h(state) -> np.ndarray``.  Pass ``None`` if providing a
            constant *observation_matrix* instead (linear sensor).
        jacobian_fn:
            ``H(state) -> np.ndarray``.  Pass ``None`` for linear sensors
            (constant *observation_matrix* is returned instead).
        observation_matrix:
            Constant H matrix for linear sensors.
        default_covariance:
            Default R matrix, shape ``(n_obs, n_obs)``.
        value_extractor:
            Reads the raw measurement vector from a DataFrame row.
        covariance_extractor:
            Optional: builds R from a row (overrides *default_covariance*).
        n_obs:
            Dimension of the observation vector (used when inferred from
            *observation_matrix* is not possible).
        reference_fn:
            ``(lat_ref, lon_ref) -> None`` called at segment start.  Useful
            when ``observe_fn`` closes over mutable geometry.
        name, sensor_tag, timestamp_col:
            See :meth:`fixed_point`.

        Examples
        --------
        Barometer reading vertical position (north component of state)::

            import numpy as np
            baro = SensorFactory.callable_sensor(
                observe_fn=None,
                observation_matrix=np.array([[0., 1., 0., 0.]]),
                default_covariance=np.array([[4.0]]),
                value_extractor=lambda row: np.array([row["altitude_m"]]),
                name="barometer",
                sensor_tag="baro",
            )
        """
        model = CallableModel(
            observe_fn=observe_fn,
            jacobian_fn=jacobian_fn,
            _n_obs=n_obs,
            default_cov=default_covariance,
            observation_matrix_val=observation_matrix,
            reference_fn=reference_fn,
        )
        return GenericSensorConfig(
            name=name,
            model=model,
            sensor_tag=sensor_tag or name,
            timestamp_col=timestamp_col,
            value_extractor=value_extractor,
            covariance_extractor=covariance_extractor,
        )


# --------------------------------------------------------------------------- #
# Register built-in factory methods under their canonical names
# --------------------------------------------------------------------------- #
SensorFactory._registry["fixed_point"] = SensorFactory.fixed_point
SensorFactory._registry["range"] = SensorFactory.range_sensor
SensorFactory._registry["bearing"] = SensorFactory.bearing_sensor
SensorFactory._registry["callable"] = SensorFactory.callable_sensor
