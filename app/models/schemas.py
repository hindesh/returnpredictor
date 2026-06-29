from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Input schemas ─────────────────────────────────────────────────────────────

class CustomerInfo(BaseModel):
    customer_id: str = Field(default_factory=lambda: f"C{uuid4().hex[:6].upper()}")
    age_days: int = Field(default=365, ge=1, description="Account age in days")
    return_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    total_orders: int = Field(default=5, ge=1)
    avg_order_value: float = Field(default=75.0, gt=0)


class ProductInfo(BaseModel):
    product_id: str = Field(default_factory=lambda: f"P{uuid4().hex[:6].upper()}")
    category: str = Field(default="Electronics")
    price: float = Field(gt=0, description="Base price in USD")
    rating: float = Field(default=4.0, ge=1.0, le=5.0)
    historical_return_rate: float = Field(default=0.15, ge=0.0, le=1.0)


class SellerInfo(BaseModel):
    seller_id: str = Field(default_factory=lambda: f"S{uuid4().hex[:4].upper()}")
    rating: float = Field(default=4.0, ge=1.0, le=5.0)
    return_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    total_orders: int = Field(default=1000, ge=1)


class OrderDetails(BaseModel):
    order_id: Optional[str] = None
    order_date: Optional[datetime] = None
    order_value: float = Field(gt=0, description="Total value after discount")
    discount_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    quantity: int = Field(default=1, ge=1, le=100)
    is_gift: bool = False
    payment_method: str = Field(default="credit_card")
    shipping_method: str = Field(default="standard")
    delivery_delay_days: int = Field(default=0)


class PredictionRequest(BaseModel):
    customer: CustomerInfo
    product: ProductInfo
    seller: SellerInfo
    order: OrderDetails

    def to_flat_record(self) -> dict:
        return {
            "customer_age_days": self.customer.age_days,
            "customer_return_rate": self.customer.return_rate,
            "customer_total_orders": self.customer.total_orders,
            "customer_avg_order_value": self.customer.avg_order_value,
            "product_category": self.product.category,
            "product_price": self.product.price,
            "product_rating": self.product.rating,
            "product_return_rate": self.product.historical_return_rate,
            "seller_rating": self.seller.rating,
            "seller_return_rate": self.seller.return_rate,
            "seller_total_orders": self.seller.total_orders,
            "order_date": self.order.order_date or datetime.utcnow(),
            "order_value": self.order.order_value,
            "discount_pct": self.order.discount_pct,
            "quantity": self.order.quantity,
            "is_gift": int(self.order.is_gift),
            "payment_method": self.order.payment_method,
            "shipping_method": self.order.shipping_method,
            "delivery_delay_days": self.order.delivery_delay_days,
        }


class BatchPredictionRequest(BaseModel):
    orders: list[PredictionRequest] = Field(min_length=1, max_length=100)


# ── Output schemas ────────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    feature: str
    description: str
    impact: float = Field(description="Absolute SHAP contribution")
    direction: Literal["increases", "decreases"]


class ModelMeta(BaseModel):
    name: str
    version: str
    trained_at: str


class PredictionResponse(BaseModel):
    order_id: str
    return_probability: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    expected_return_cost: float = Field(description="Estimated return cost in USD")
    top_risk_factors: list[RiskFactor]
    recommendations: list[str]
    model_info: ModelMeta
    processed_at: datetime


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    processed_at: datetime


# ── Order CRUD schemas ────────────────────────────────────────────────────────

class OrderCreate(PredictionRequest):
    pass


class OrderRecord(BaseModel):
    order_id: str
    customer_id: str
    product_id: str
    seller_id: str
    order_value: float
    product_category: str
    created_at: datetime
    prediction: Optional[PredictionResponse] = None

    model_config = {"from_attributes": True}
