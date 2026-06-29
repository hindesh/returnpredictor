from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.explainer import explainer
from app.ml.predictor import predictor
from app.models.orm import Order, Prediction
from app.models.schemas import (
    OrderCreate,
    OrderRecord,
    PredictionResponse,
    ModelMeta,
)

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("", response_model=OrderRecord, status_code=201, summary="Submit order and get return risk prediction")
def create_order(payload: OrderCreate, db: Session = Depends(get_db)):
    if not predictor._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")

    order_id = payload.order.order_id or f"ORD-{uuid4().hex[:8].upper()}"

    # Persist order
    order = Order(
        id=order_id,
        customer_id=payload.customer.customer_id,
        product_id=payload.product.product_id,
        seller_id=payload.seller.seller_id,
        customer_age_days=payload.customer.age_days,
        customer_return_rate=payload.customer.return_rate,
        customer_total_orders=payload.customer.total_orders,
        customer_avg_order_value=payload.customer.avg_order_value,
        product_category=payload.product.category,
        product_price=payload.product.price,
        product_rating=payload.product.rating,
        product_return_rate=payload.product.historical_return_rate,
        seller_rating=payload.seller.rating,
        seller_return_rate=payload.seller.return_rate,
        seller_total_orders=payload.seller.total_orders,
        order_date=payload.order.order_date or datetime.utcnow(),
        order_value=payload.order.order_value,
        discount_pct=payload.order.discount_pct,
        quantity=payload.order.quantity,
        is_gift=payload.order.is_gift,
        payment_method=payload.order.payment_method,
        shipping_method=payload.order.shipping_method,
        delivery_delay_days=payload.order.delivery_delay_days,
    )
    db.add(order)
    db.flush()

    # Run prediction
    record = payload.to_flat_record()
    prob, X = predictor.predict(record)
    pred_response: PredictionResponse = explainer.explain(
        X=X,
        prob=prob,
        order_value=float(payload.order.order_value),
        order_id=order_id,
        model_meta=predictor.meta,
    )

    # Persist prediction
    pred_orm = Prediction(
        order_id=order_id,
        return_probability=pred_response.return_probability,
        risk_level=pred_response.risk_level,
        expected_return_cost=pred_response.expected_return_cost,
        top_risk_factors=[f.model_dump() for f in pred_response.top_risk_factors],
        recommendations=pred_response.recommendations,
        model_name=predictor.meta["name"],
        model_version=predictor.meta["version"],
    )
    db.add(pred_orm)
    db.commit()

    return OrderRecord(
        order_id=order_id,
        customer_id=payload.customer.customer_id,
        product_id=payload.product.product_id,
        seller_id=payload.seller.seller_id,
        order_value=float(payload.order.order_value),
        product_category=payload.product.category,
        created_at=order.order_date or datetime.utcnow(),
        prediction=pred_response,
    )


@router.get("", response_model=list[OrderRecord], summary="List recent orders with predictions")
def list_orders(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    orders = db.query(Order).order_by(Order.created_at.desc()).offset(offset).limit(limit).all()
    result = []
    for o in orders:
        pred_orm = (
            db.query(Prediction)
            .filter(Prediction.order_id == o.id)
            .order_by(Prediction.created_at.desc())
            .first()
        )
        pred_resp = _orm_to_pred_response(pred_orm) if pred_orm else None
        result.append(OrderRecord(
            order_id=o.id,
            customer_id=o.customer_id,
            product_id=o.product_id,
            seller_id=o.seller_id,
            order_value=o.order_value or 0.0,
            product_category=o.product_category or "Unknown",
            created_at=o.created_at,
            prediction=pred_resp,
        ))
    return result


@router.get("/{order_id}", response_model=OrderRecord, summary="Get a specific order with its prediction")
def get_order(order_id: str, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail=f"Order '{order_id}' not found")

    pred_orm = (
        db.query(Prediction)
        .filter(Prediction.order_id == order_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )
    return OrderRecord(
        order_id=order.id,
        customer_id=order.customer_id,
        product_id=order.product_id,
        seller_id=order.seller_id,
        order_value=order.order_value or 0.0,
        product_category=order.product_category or "Unknown",
        created_at=order.created_at,
        prediction=_orm_to_pred_response(pred_orm) if pred_orm else None,
    )


def _orm_to_pred_response(pred: Prediction) -> PredictionResponse:
    from app.models.schemas import RiskFactor
    factors = [RiskFactor(**f) for f in (pred.top_risk_factors or [])]
    return PredictionResponse(
        order_id=pred.order_id,
        return_probability=pred.return_probability,
        risk_level=pred.risk_level,
        expected_return_cost=pred.expected_return_cost,
        top_risk_factors=factors,
        recommendations=pred.recommendations or [],
        model_info=ModelMeta(
            name=pred.model_name or "Unknown",
            version=pred.model_version or "1.0.0",
            trained_at="",
        ),
        processed_at=pred.created_at,
    )
