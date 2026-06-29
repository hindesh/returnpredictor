from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

CATEGORICAL_COLS = ['product_category', 'payment_method', 'shipping_method']

KNOWN_CATEGORIES = {
    'product_category': [
        'Electronics', 'Clothing', 'Shoes', 'Books', 'Home & Garden',
        'Beauty', 'Furniture', 'Sports', 'Toys', 'Food', 'Unknown',
    ],
    'payment_method': [
        'credit_card', 'debit_card', 'paypal', 'buy_now_pay_later',
        'bank_transfer', 'Unknown',
    ],
    'shipping_method': ['standard', 'express', 'overnight', 'economy', 'Unknown'],
}

FEATURE_NAMES = [
    # Customer (5)
    'customer_age_days',
    'customer_return_rate',
    'customer_total_orders',
    'customer_avg_order_value',
    'is_new_customer',
    # Product (5)
    'product_price',
    'product_rating',
    'product_return_rate',
    'product_category',
    'is_high_value_product',
    # Seller (3)
    'seller_rating',
    'seller_return_rate',
    'seller_total_orders',
    # Order (8)
    'order_value',
    'discount_pct',
    'quantity',
    'is_gift',
    'payment_method',
    'shipping_method',
    'delivery_delay_days',
    'is_late_delivery',
    # Temporal (5)
    'order_day_of_week',
    'order_month',
    'order_is_weekend',
    'order_is_holiday_season',
    'order_is_january',
    # Interactions (4)
    'discount_x_return_rate',
    'delay_x_rating',
    'customer_rate_x_product_rate',
    'value_x_discount',
]

FEATURE_DESCRIPTIONS = {
    'customer_age_days': 'Customer account age (days)',
    'customer_return_rate': 'Customer historical return rate',
    'customer_total_orders': 'Customer total orders placed',
    'customer_avg_order_value': 'Customer average order value',
    'is_new_customer': 'New customer (≤2 orders)',
    'product_price': 'Product base price',
    'product_rating': 'Product average rating',
    'product_return_rate': 'Product historical return rate',
    'product_category': 'Product category',
    'is_high_value_product': 'High-value product (>$100)',
    'seller_rating': 'Seller average rating',
    'seller_return_rate': 'Seller historical return rate',
    'seller_total_orders': 'Seller total orders fulfilled',
    'order_value': 'Order total value after discount',
    'discount_pct': 'Discount percentage applied',
    'quantity': 'Number of items ordered',
    'is_gift': 'Gift purchase',
    'payment_method': 'Payment method used',
    'shipping_method': 'Shipping method selected',
    'delivery_delay_days': 'Delivery delay (days)',
    'is_late_delivery': 'Order delivered late',
    'order_day_of_week': 'Day of week ordered (0=Mon)',
    'order_month': 'Month of order',
    'order_is_weekend': 'Weekend order',
    'order_is_holiday_season': 'Holiday season (Nov–Dec)',
    'order_is_january': 'January order (post-holiday returns spike)',
    'discount_x_return_rate': 'Discount × product return rate interaction',
    'delay_x_rating': 'Delivery delay × (5 – product rating) interaction',
    'customer_rate_x_product_rate': 'Customer × product return rate interaction',
    'value_x_discount': 'Order value × discount interaction',
}


class FeatureEngineer:
    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            values = list(df[col].astype(str).unique()) + ['Unknown']
            le.fit(values)
            self.label_encoders[col] = le
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        return self._compute(df.copy()).values.astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def transform_single(self, record: dict) -> np.ndarray:
        return self.transform(pd.DataFrame([record]))

    def _encode(self, series: pd.Series, col: str) -> pd.Series:
        le = self.label_encoders[col]
        known = set(le.classes_)

        def _safe(x: str) -> int:
            return le.transform([x])[0] if x in known else le.transform(['Unknown'])[0]

        return series.astype(str).apply(_safe)

    def _compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'order_date' in df.columns:
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        else:
            df['order_date'] = pd.Timestamp.now()
        df['order_date'] = df['order_date'].fillna(pd.Timestamp.now())

        r = pd.DataFrame(index=df.index)

        # Customer
        r['customer_age_days'] = df['customer_age_days'].fillna(365).astype(float)
        r['customer_return_rate'] = df['customer_return_rate'].fillna(0.10).astype(float)
        r['customer_total_orders'] = df['customer_total_orders'].fillna(1).astype(float)
        r['customer_avg_order_value'] = df['customer_avg_order_value'].fillna(50.0).astype(float)
        r['is_new_customer'] = (df['customer_total_orders'].fillna(1) <= 2).astype(float)

        # Product
        r['product_price'] = df['product_price'].fillna(50.0).astype(float)
        r['product_rating'] = df['product_rating'].fillna(3.5).astype(float)
        r['product_return_rate'] = df['product_return_rate'].fillna(0.15).astype(float)
        r['product_category'] = self._encode(df['product_category'].fillna('Unknown'), 'product_category')
        r['is_high_value_product'] = (df['product_price'].fillna(0) > 100).astype(float)

        # Seller
        r['seller_rating'] = df['seller_rating'].fillna(4.0).astype(float)
        r['seller_return_rate'] = df['seller_return_rate'].fillna(0.15).astype(float)
        r['seller_total_orders'] = df['seller_total_orders'].fillna(100).astype(float)

        # Order
        r['order_value'] = df['order_value'].fillna(df.get('product_price', 50.0)).astype(float)
        r['discount_pct'] = df['discount_pct'].fillna(0).astype(float)
        r['quantity'] = df['quantity'].fillna(1).astype(float)
        r['is_gift'] = df['is_gift'].fillna(0).astype(float)
        r['payment_method'] = self._encode(df['payment_method'].fillna('Unknown'), 'payment_method')
        r['shipping_method'] = self._encode(df['shipping_method'].fillna('Unknown'), 'shipping_method')
        r['delivery_delay_days'] = df['delivery_delay_days'].fillna(0).astype(float)
        r['is_late_delivery'] = (df['delivery_delay_days'].fillna(0) > 0).astype(float)

        # Temporal
        r['order_day_of_week'] = df['order_date'].dt.dayofweek.astype(float)
        r['order_month'] = df['order_date'].dt.month.astype(float)
        r['order_is_weekend'] = (df['order_date'].dt.dayofweek >= 5).astype(float)
        r['order_is_holiday_season'] = df['order_date'].dt.month.isin([11, 12]).astype(float)
        r['order_is_january'] = (df['order_date'].dt.month == 1).astype(float)

        # Interactions
        r['discount_x_return_rate'] = r['discount_pct'] * r['product_return_rate']
        r['delay_x_rating'] = r['delivery_delay_days'] * (5.0 - r['product_rating'])
        r['customer_rate_x_product_rate'] = r['customer_return_rate'] * r['product_return_rate']
        r['value_x_discount'] = r['order_value'] * r['discount_pct'] / 100.0

        return r[FEATURE_NAMES]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> FeatureEngineer:
        return joblib.load(path)
