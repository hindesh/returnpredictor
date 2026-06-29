from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.ml.explainer import explainer
from app.ml.predictor import predictor
from app.models.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("", response_model=PredictionResponse, summary="Predict return risk for a single order")
def predict_single(request: PredictionRequest) -> PredictionResponse:
    _require_model()
    order_id = request.order.order_id or f"ORD-{uuid4().hex[:8].upper()}"
    record = request.to_flat_record()

    try:
        prob, X = predictor.predict(record)
        return explainer.explain(
            X=X,
            prob=prob,
            order_value=float(request.order.order_value),
            order_id=order_id,
            model_meta=predictor.meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict return risk for up to 100 orders",
)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    _require_model()
    records = [r.to_flat_record() for r in request.orders]
    order_ids = [
        r.order.order_id or f"ORD-{uuid4().hex[:8].upper()}"
        for r in request.orders
    ]
    order_values = [float(r.order.order_value) for r in request.orders]

    try:
        probs, X = predictor.predict_batch(records)
        predictions = explainer.explain_batch(
            X=X,
            probs=probs,
            order_values=order_values,
            order_ids=order_ids,
            model_meta=predictor.meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        processed_at=datetime.utcnow(),
    )


def _require_model() -> None:
    if not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train_model.py` first.",
        )
