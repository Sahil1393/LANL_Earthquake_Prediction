import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from src.exception import CustomException
from src.logger import logging
from src.Pipeline.predict_pipeline import PredictPipeline


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    acoustic_data: List[float] = Field(
        ...,
        description="List of acoustic signal values (minimum 100 values)."
    )

    @field_validator("acoustic_data")
    @classmethod
    def validate_length(cls, v):
        if len(v) < 100:
            raise ValueError(
                f"acoustic_data must contain at least 100 values, got {len(v)}."
            )
        return v


class PredictResponse(BaseModel):
    time_to_failure_seconds: float
    input_length: int
    model_folds: int


class BatchPredictRequest(BaseModel):
    segments: List[List[float]] = Field(
        ...,
        description="List of acoustic data segments. Each must have at least 100 values."
    )

    @field_validator("segments")
    @classmethod
    def validate_segments(cls, v):
        if len(v) == 0:
            raise ValueError("segments list must not be empty.")
        for i, seg in enumerate(v):
            if len(seg) < 100:
                raise ValueError(
                    f"Segment at index {i} must have at least 100 values, got {len(seg)}."
                )
        return v


class BatchPredictResponse(BaseModel):
    predictions: List[float]
    total_segments: int
    model_folds: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_type: str
    n_folds: int
    cv_mae: float
    n_features: int
    feature_names: List[str]


# ─────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────

_start_time = time.time()
_predict_pipeline: PredictPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predict_pipeline
    logging.info("FastAPI startup: initialising prediction pipeline...")
    try:
        _predict_pipeline = PredictPipeline()
        if os.path.exists(_predict_pipeline.model_path):
            from src.utils import load_object
            load_object(_predict_pipeline.model_path)
            logging.info("Model loaded successfully at startup.")
        else:
            logging.warning(
                f"Model file not found at {_predict_pipeline.model_path}. "
                "Run the training pipeline first."
            )
    except Exception as e:
        logging.warning(f"Startup warning: {e}")
        _predict_pipeline = PredictPipeline()
    yield
    logging.info("FastAPI shutdown.")


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title="LANL Earthquake Prediction API",
    description=(
        "Predicts **time remaining before the next earthquake** (seconds) "
        "from raw seismic acoustic signal data using a LightGBM ensemble "
        "trained with 5-fold cross-validation.\n\n"
        "**Model**: LightGBM · 5-Fold KFold · StandardScaler · CV MAE ≈ 0.106"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────
# Global exception handler — shows real error
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = traceback.format_exc()
    logging.error(f"Unhandled exception on {request.url}:\n{error_detail}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": error_detail}
    )


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _get_pipeline() -> PredictPipeline:
    if _predict_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first."
        )
    if not os.path.exists(_predict_pipeline.model_path):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model file not found at '{_predict_pipeline.model_path}'. "
                "Run: python -m src.Pipeline.train_pipeline"
            )
        )
    return _predict_pipeline


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Web UI"
)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    summary="Health check"
)
async def health():
    model_path = "artifacts/final_model.pkl"
    return HealthResponse(
        status="ok",
        model_loaded=os.path.exists(model_path),
        model_path=model_path,
        uptime_seconds=round(time.time() - _start_time, 2)
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict time_to_failure from a single acoustic data segment"
)
async def predict(body: PredictRequest):
    """
    Send a list of acoustic signal values and receive the predicted
    **time remaining before the next earthquake** in seconds.

    - Minimum **100 values** required (150,000 recommended).
    - Prediction is clipped to **[0, 16]** seconds.
    """
    pipeline = _get_pipeline()

    try:
        prediction = pipeline.predict(body.acoustic_data)

        from src.utils import load_object
        model_object = load_object(pipeline.model_path)
        n_folds = len(model_object["models"])

        logging.info(
            f"/predict | input_length={len(body.acoustic_data)} "
            f"| prediction={prediction:.4f}s"
        )

        return PredictResponse(
            time_to_failure_seconds=float(prediction),
            input_length=len(body.acoustic_data),
            model_folds=n_folds
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/predict error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
    summary="Predict time_to_failure for multiple segments at once"
)
async def predict_batch(body: BatchPredictRequest):
    """
    Send multiple acoustic data segments and receive a prediction for each.

    - Each segment must have at least **100 values**.
    - Predictions are clipped to **[0, 16]** seconds.
    """
    pipeline = _get_pipeline()

    try:
        predictions = []
        for segment in body.segments:
            pred = pipeline.predict(segment)
            predictions.append(float(pred))

        from src.utils import load_object
        model_object = load_object(pipeline.model_path)
        n_folds = len(model_object["models"])

        logging.info(
            f"/predict/batch | segments={len(body.segments)} "
            f"| predictions={[round(p, 4) for p in predictions]}"
        )

        return BatchPredictResponse(
            predictions=predictions,
            total_segments=len(predictions),
            model_folds=n_folds
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/predict/batch error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get metadata about the trained model"
)
async def model_info():
    model_path = "artifacts/final_model.pkl"

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model file not found at '{model_path}'. "
                "Run: python -m src.Pipeline.train_pipeline"
            )
        )

    try:
        from src.utils import load_object
        model_object = load_object(model_path)

        return ModelInfoResponse(
            model_type="LightGBM Ensemble (KFold)",
            n_folds=len(model_object["models"]),
            cv_mae=round(float(model_object["cv_mae"]), 6),
            n_features=len(model_object["feature_columns"]),
            feature_names=model_object["feature_columns"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"/model/info error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
