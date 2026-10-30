"""FastAPI service for live house-price predictions."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from house_price.predict import load_model, predict_record

logger = logging.getLogger(__name__)


def create_app(model: Any | None = None) -> FastAPI:
    """Create the API and load the saved model once when the service starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model if model is not None else load_model()
        logger.info("House-price API model is ready")
        yield

    application = FastAPI(
        title="House Price Prediction API",
        description="Predict Ames Housing sale prices using the packaged model.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request) -> dict[str, str]:
        """Report service readiness after the model has loaded."""
        if not hasattr(request.app.state, "model"):
            raise HTTPException(status_code=503, detail="Model is not ready.")
        return {"status": "ok"}

    @application.post("/predict")
    def predict(payload: dict[str, Any], request: Request) -> dict[str, float]:
        """Predict the sale price for one JSON object of raw house features."""
        try:
            price = predict_record(payload, request.app.state.model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"predicted_sale_price": price}

    return application


app = create_app()
