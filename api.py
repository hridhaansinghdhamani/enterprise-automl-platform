"""
Enterprise AutoML FastAPI
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.components.model_monitor import ModelMonitor
from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(
    title="Enterprise AutoML API",
    description="Enterprise AutoML Prediction API",
    version="2.0.0",
)

pipeline = PredictionPipeline()
monitor = ModelMonitor()


class PredictionRequest(BaseModel):
    data: list[dict[str, Any]]


@app.get("/")
def home():
    return {
        "application": "Enterprise AutoML Platform",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/monitoring")
def monitoring():
    return monitor.summary()


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        dataframe = pd.DataFrame(request.data)

        result = pipeline.predict(dataframe)

        return {
            "status": "success",
            "total_records": len(result),
            "prediction": result.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.get("/model-info")
def model_info():
    return {
        "framework": "Scikit-Learn",
        "prediction_type": "Classification / Regression",
        "supports_batch_prediction": True,
        "supports_monitoring": True,
        "supports_versioning": True,
    }