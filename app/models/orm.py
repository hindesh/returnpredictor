from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    customer_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String, nullable=False)
    seller_id: Mapped[str] = mapped_column(String, nullable=False)

    # Customer
    customer_age_days: Mapped[int | None] = mapped_column(Integer)
    customer_return_rate: Mapped[float | None] = mapped_column(Float)
    customer_total_orders: Mapped[int | None] = mapped_column(Integer)
    customer_avg_order_value: Mapped[float | None] = mapped_column(Float)

    # Product
    product_category: Mapped[str | None] = mapped_column(String)
    product_price: Mapped[float | None] = mapped_column(Float)
    product_rating: Mapped[float | None] = mapped_column(Float)
    product_return_rate: Mapped[float | None] = mapped_column(Float)

    # Seller
    seller_rating: Mapped[float | None] = mapped_column(Float)
    seller_return_rate: Mapped[float | None] = mapped_column(Float)
    seller_total_orders: Mapped[int | None] = mapped_column(Integer)

    # Order
    order_date: Mapped[datetime | None] = mapped_column(DateTime)
    order_value: Mapped[float | None] = mapped_column(Float)
    discount_pct: Mapped[float | None] = mapped_column(Float)
    quantity: Mapped[int | None] = mapped_column(Integer)
    is_gift: Mapped[bool] = mapped_column(Boolean, default=False)
    payment_method: Mapped[str | None] = mapped_column(String)
    shipping_method: Mapped[str | None] = mapped_column(String)
    delivery_delay_days: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    return_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String)
    expected_return_cost: Mapped[float] = mapped_column(Float)
    top_risk_factors: Mapped[list | None] = mapped_column(JSON)
    recommendations: Mapped[list | None] = mapped_column(JSON)
    model_name: Mapped[str | None] = mapped_column(String)
    model_version: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
