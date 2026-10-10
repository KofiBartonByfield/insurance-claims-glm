"""
models.py

Three claim-frequency models, in increasing order of complexity:

1. Naive baseline              — one overall claim rate applied to every policy.
2. Poisson GLM                 — the industry-standard actuarial pricing model.
3. XGBoost (Poisson objective) — a flexible benchmark, to see how much a
                                  non-linear model actually buys over the GLM.

Exposure (fraction of a year the policy was on-risk) is used as an offset/
weight throughout, so every model is fit on a genuine annualised claim
rate rather than raw counts.
"""

import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor


class NaiveBaseline:
    """Predicts the same overall claim frequency for every policy."""

    def fit(self, y, exposure):
        self.rate_ = np.sum(y) / np.sum(exposure)
        return self

    def predict(self, exposure):
        return self.rate_ * exposure


def fit_poisson_glm(X_train, y_train, exposure_train, alpha=1e-4):
    """Poisson GLM with exposure as a sample weight (log link, rate target)."""
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(X_train, y_train / exposure_train, sample_weight=exposure_train)
    return model


def fit_xgboost_poisson(X_train, y_train, exposure_train, **kwargs):
    """
    XGBoost with a Poisson objective. Exposure is passed as base_margin
    (log-offset) so predictions come out already rate-scaled, matching the
    GLM's target definition.
    """
    params = dict(
        objective="count:poisson",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        enable_categorical=True,
        # base_score is ignored in training when base_margin is given, but IS added
        # back at predict() time. Auto-estimated it shrinks every rate prediction
        # ~20x; 1.0 (log = 0) makes predict() return the true rate.
        base_score=1.0,
        random_state=42,
    )
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, base_margin=np.log(exposure_train), verbose=False)
    return model
