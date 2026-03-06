# kalmangps

Kalman filter pipeline for GPS trajectories with variable accuracy and multi-sensor fusion support.

Built with [uv](https://github.com/astral-sh/uv) and [hatch](https://hatch.pypa.io/).

## Features

- **Trajectory splitting** by an arbitrary user column (trip ID, device ID, etc.)
- **Time-gap splitting** — segments restart when the gap between fixes is too large
- **Per-measurement Kalman filtering** — GPS accuracy is used as a per-step observation covariance (variable R), so a 3 m fix is trusted much more than a 50 m fix
- **Local Cartesian projection** — works in metres (East/North), not degrees, so the state model is physically meaningful
- **Multi-sensor fusion ready** — the `Measurement` / `SensorConfig` abstractions and the `SensorFuser` stub are in place for GPS + IMU fusion in the next phase

## Quick start

```python
import pandas as pd
from kalmangps import GPSKalmanPipeline, GPSConfig

df = pd.read_csv("gps_log.csv")

pipeline = GPSKalmanPipeline(
    gps_config=GPSConfig(
        lat_col="lat",
        lon_col="lon",
        accuracy_col="h_accuracy",   # per-fix 1-sigma accuracy in metres
        timestamp_col="utc_time",
    ),
    group_col="trip_id",             # split by this column first
    max_time_gap="5 min",            # then split on time gaps > 5 min
)

result = pipeline.run(df)
smoothed = result.to_dataframe()
# New columns: filtered_lat, filtered_lon, filtered_vx_ms, filtered_vy_ms,
#              pos_uncertainty_x_m, pos_uncertainty_y_m
```

## Installation

```bash
uv pip install kalmangps
```

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
hatch run test      # run tests with coverage
hatch run lint      # ruff lint
hatch run fmt       # ruff format
```

## Architecture

```
kalmangps/
├── sensors/        # Measurement + SensorConfig base classes; GPSConfig
├── state/          # StateSpaceModel; ConstantVelocity2D (F, Q matrices)
├── filters/        # VariableNoiseKalmanFilter (per-step R support)
├── pipeline/       # TrajectorySplitter + GPSKalmanPipeline
└── fusion/         # SensorFuser stub (multi-sensor, next phase)
```

### Adding a new sensor (next phase)

1. Subclass `SensorConfig` in `sensors/` — provide `observation_matrix` (H) and `measurement_from_row`.
2. Register with `SensorFuser.register_sensor(name, config)`.
3. Feed measurements into `SensorFuser.run(stream)`.

The shared state vector `[x, y, vx, vy]` and the Kalman update step do not change.
