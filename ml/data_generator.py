from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_data(n_samples: int = 50_000, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic e-commerce order data for return prediction."""
    rng = np.random.default_rng(random_seed)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 6, 1)
    date_range_days = (end_date - start_date).days

    # ── Customers ──────────────────────────────────────────────────────────────
    n_customers = 5_000
    cust_age = (rng.exponential(500, n_customers) + 30).astype(int)
    cust_return_rate = rng.beta(2, 8, n_customers)
    cust_total_orders = (rng.negative_binomial(5, 0.4, n_customers) + 1).astype(int)
    cust_avg_order_val = rng.lognormal(4.0, 0.8, n_customers)

    # ── Products ───────────────────────────────────────────────────────────────
    n_products = 1_000
    categories = [
        'Electronics', 'Clothing', 'Shoes', 'Books', 'Home & Garden',
        'Beauty', 'Furniture', 'Sports', 'Toys', 'Food',
    ]
    cat_base_return = {
        'Electronics': 0.15, 'Clothing': 0.30, 'Shoes': 0.35, 'Books': 0.05,
        'Home & Garden': 0.10, 'Beauty': 0.08, 'Furniture': 0.12, 'Sports': 0.12,
        'Toys': 0.08, 'Food': 0.03,
    }
    prod_cats = rng.choice(categories, n_products)
    prod_prices = np.clip(rng.lognormal(3.5, 1.0, n_products), 5.0, 2_000.0)
    prod_ratings = np.clip(rng.normal(3.8, 0.7, n_products), 1.0, 5.0)
    prod_return_rates = np.array([
        np.clip(cat_base_return[c] * rng.uniform(0.5, 1.5), 0.01, 0.80)
        for c in prod_cats
    ])

    # ── Sellers ────────────────────────────────────────────────────────────────
    n_sellers = 200
    seller_ratings = np.clip(rng.normal(4.0, 0.5, n_sellers), 1.0, 5.0)
    seller_return_rates = rng.beta(2, 8, n_sellers)
    seller_total_orders = (rng.negative_binomial(10, 0.2, n_sellers) * 100 + 100).astype(int)
    prod_seller_idx = rng.integers(0, n_sellers, n_products)

    # ── Orders ─────────────────────────────────────────────────────────────────
    cust_idx = rng.integers(0, n_customers, n_samples)
    prod_idx = rng.integers(0, n_products, n_samples)
    seller_idx = prod_seller_idx[prod_idx]

    order_dates = [
        start_date + timedelta(days=int(d))
        for d in rng.integers(0, date_range_days, n_samples)
    ]

    discount_choices = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    discount_probs = [0.35, 0.10, 0.15, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03]
    discounts = rng.choice(discount_choices, n_samples, p=discount_probs).astype(float)

    quantities = rng.choice([1, 2, 3, 4, 5], n_samples, p=[0.60, 0.20, 0.10, 0.06, 0.04])
    is_gift = rng.choice([0, 1], n_samples, p=[0.85, 0.15])

    payment_methods = rng.choice(
        ['credit_card', 'debit_card', 'paypal', 'buy_now_pay_later', 'bank_transfer'],
        n_samples, p=[0.40, 0.25, 0.20, 0.10, 0.05],
    )
    shipping_methods = rng.choice(
        ['standard', 'express', 'overnight', 'economy'],
        n_samples, p=[0.50, 0.25, 0.10, 0.15],
    )

    delivery_delays = np.round(rng.normal(0, 2, n_samples)).astype(int)
    delivery_delays = np.where(shipping_methods == 'economy', delivery_delays + 1, delivery_delays)

    base_prices = prod_prices[prod_idx]
    order_values = base_prices * (1 - discounts / 100.0) * quantities

    # ── Return probability (latent signal) ─────────────────────────────────────
    p = np.zeros(n_samples)
    p += prod_return_rates[prod_idx]
    p += cust_return_rate[cust_idx] * 0.50
    p += seller_return_rates[seller_idx] * 0.20
    p += (discounts / 100.0) * 0.15
    p += np.clip(delivery_delays, 0, 10) * 0.02
    p -= is_gift * 0.05
    p += np.where(prod_ratings[prod_idx] < 3.0, 0.10, 0.0)
    p += np.where(base_prices > 200, 0.05, 0.0)
    p += np.where(payment_methods == 'buy_now_pay_later', 0.08, 0.0)
    p += np.array([0.05 if d.month == 1 else 0.0 for d in order_dates])
    p = np.clip(p, 0.01, 0.95)

    is_returned = rng.binomial(1, p)

    return pd.DataFrame({
        'order_id': [f'ORD{i:07d}' for i in range(n_samples)],
        'customer_id': [f'C{cust_idx[i]:05d}' for i in range(n_samples)],
        'product_id': [f'P{prod_idx[i]:05d}' for i in range(n_samples)],
        'seller_id': [f'S{seller_idx[i]:04d}' for i in range(n_samples)],
        # Customer
        'customer_age_days': cust_age[cust_idx],
        'customer_return_rate': cust_return_rate[cust_idx],
        'customer_total_orders': cust_total_orders[cust_idx],
        'customer_avg_order_value': cust_avg_order_val[cust_idx],
        # Product
        'product_category': prod_cats[prod_idx],
        'product_price': base_prices,
        'product_rating': prod_ratings[prod_idx],
        'product_return_rate': prod_return_rates[prod_idx],
        # Seller
        'seller_rating': seller_ratings[seller_idx],
        'seller_return_rate': seller_return_rates[seller_idx],
        'seller_total_orders': seller_total_orders[seller_idx],
        # Order
        'order_date': order_dates,
        'order_value': order_values,
        'discount_pct': discounts,
        'quantity': quantities,
        'is_gift': is_gift,
        'payment_method': payment_methods,
        'shipping_method': shipping_methods,
        'delivery_delay_days': delivery_delays,
        # Target
        'is_returned': is_returned,
    })
