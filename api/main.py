"""FastAPI backend for kalmangps — upload a CSV and run the pipeline."""

from __future__ import annotations

import io
import json

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from kalmangps import GPSConfig, GPSKalmanPipeline

app = FastAPI(title="kalmangps")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/run")
async def run_pipeline(
    file: UploadFile = File(...),
    config: str = Form("{}"),
):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc

    try:
        cfg = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid config JSON: {exc}") from exc

    # Drop rows flagged as outliers before filtering
    outlier_col = cfg.get("outlier_col") or None
    if outlier_col and outlier_col in df.columns:
        df = df[~df[outlier_col].astype(str).str.upper().isin({"TRUE", "1", "YES"})]

    gps_config = GPSConfig(
        lat_col=cfg.get("lat_col", "lat"),
        lon_col=cfg.get("lon_col", "lon"),
        accuracy_col=cfg.get("accuracy_col") or None,
        timestamp_col=cfg.get("timestamp_col", "timestamp"),
        default_accuracy_m=float(cfg.get("default_accuracy_m", 10.0)),
    )

    pipeline = GPSKalmanPipeline(
        gps_config=gps_config,
        group_col=cfg.get("group_col") or None,
        max_time_gap=f"{cfg.get('max_time_gap', 300)}s",
        min_segment_length=int(cfg.get("min_segment_length", 3)),
    )

    try:
        result = pipeline.run(df)
    except Exception as exc:
        raise HTTPException(422, f"Pipeline error: {exc}") from exc

    out = result.to_dataframe()
    # Replace NaN/NaT/inf/-inf with None so JSON serialisation doesn't break
    out = out.replace([float('inf'), float('-inf')], None)
    out = out.where(out.notna(), other=None)

    return {
        "records": out.to_dict(orient="records"),
        "n_segments": result.n_segments,
        "n_points": result.n_points,
        "lat_col": gps_config.lat_col,
        "lon_col": gps_config.lon_col,
    }
