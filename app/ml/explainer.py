from __future__ import annotations
from datetime import datetime
from typing import Any

import numpy as np
import shap

from ml.feature_engineering import FEATURE_DESCRIPTIONS, FEATURE_NAMES
from app.models.schemas import ModelMeta, PredictionResponse, RiskFactor


_FEATURE_RECS: dict[str, str] = {
    "customer_return_rate": "Include a personalised thank-you note — this customer returns frequently.",
    "product_return_rate": "Add a detailed product-fit / setup guide to reduce misuse-driven returns.",
    "delivery_delay_days": "Proactively notify the customer of the delay and offer a discount on next order.",
    "is_late_delivery": "Reach out proactively — late deliveries significantly raise return risk.",
    "discount_pct": "High discount suggests impulse purchase; include product-value messaging.",
    "product_rating": "Low-rated product — enclose support contact details and easy-exchange instructions.",
    "is_new_customer": "New customer onboarding guide and hassle-free return policy reduces anxiety.",
    "payment_method": "BNPL customers return more often — add payment-plan satisfaction messaging.",
    "order_is_january": "Post-holiday period — include gift-exchange guide and clear return window.",
    "customer_rate_x_product_rate": "Both customer and product have elevated return history — flag for review.",
    "discount_x_return_rate": "High discount on a return-prone product — consider manual pre-ship check.",
}


class ReturnExplainer:
    def __init__(self) -> None:
        self._explainer: shap.TreeExplainer | None = None

    def build(self, model: Any) -> None:
        self._explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X: np.ndarray,
        prob: float,
        order_value: float,
        order_id: str,
        model_meta: dict,
    ) -> PredictionResponse:
        shap_vals = self._explainer.shap_values(X)          # (1, n_features)
        if isinstance(shap_vals, list):                      # some models return list
            shap_vals = shap_vals[1]
        sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

        risk_factors = _build_risk_factors(sv)
        recs = _generate_recommendations(prob, risk_factors)
        risk_level = _risk_level(prob)
        expected_cost = round(prob * (order_value * 0.20 + 10.0), 2)

        return PredictionResponse(
            order_id=order_id,
            return_probability=round(prob, 4),
            risk_level=risk_level,
            expected_return_cost=expected_cost,
            top_risk_factors=risk_factors,
            recommendations=recs,
            model_info=ModelMeta(**model_meta),
            processed_at=datetime.utcnow(),
        )

    def explain_batch(
        self,
        X: np.ndarray,
        probs: np.ndarray,
        order_values: list[float],
        order_ids: list[str],
        model_meta: dict,
    ) -> list[PredictionResponse]:
        shap_vals = self._explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        results = []
        for i, (prob, ov, oid) in enumerate(zip(probs, order_values, order_ids)):
            sv = shap_vals[i]
            risk_factors = _build_risk_factors(sv)
            recs = _generate_recommendations(float(prob), risk_factors)
            results.append(PredictionResponse(
                order_id=oid,
                return_probability=round(float(prob), 4),
                risk_level=_risk_level(float(prob)),
                expected_return_cost=round(float(prob) * (ov * 0.20 + 10.0), 2),
                top_risk_factors=risk_factors,
                recommendations=recs,
                model_info=ModelMeta(**model_meta),
                processed_at=datetime.utcnow(),
            ))
        return results


def _risk_level(prob: float) -> str:
    if prob >= 0.60:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


def _build_risk_factors(shap_values: np.ndarray, top_n: int = 5) -> list[RiskFactor]:
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]
    factors = []
    for i in order:
        name = FEATURE_NAMES[i]
        val = float(shap_values[i])
        factors.append(RiskFactor(
            feature=name,
            description=FEATURE_DESCRIPTIONS.get(name, name),
            impact=round(abs(val), 4),
            direction="increases" if val > 0 else "decreases",
        ))
    return factors


def _generate_recommendations(
    prob: float, risk_factors: list[RiskFactor]
) -> list[str]:
    recs: list[str] = []

    if prob >= 0.60:
        recs.append("Flag for manual review before processing — high return risk detected.")
        recs.append("Require signature on delivery to confirm receipt.")
    elif prob >= 0.30:
        recs.append("Moderate risk — send proactive shipment tracking and satisfaction check-in.")
    else:
        recs.append("Low risk — standard fulfillment recommended.")

    for factor in risk_factors[:3]:
        if factor.direction == "increases":
            rec = _FEATURE_RECS.get(factor.feature)
            if rec and rec not in recs:
                recs.append(rec)

    return recs[:4]


explainer = ReturnExplainer()
