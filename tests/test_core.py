"""Offline tests on small synthetic data. Run: pytest"""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_model
from src.features import (
    CATEGORICAL_FEATURES, apply_category_levels, encode_for_glm,
    encode_for_xgboost, get_category_levels, prepare_features,
)
from src.models import NaiveBaseline, fit_poisson_glm, fit_xgboost_poisson


@pytest.fixture(scope="module")
def policies():
    """5000 policies; true claim rate is 0.05, or 0.20 for young drivers."""
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "Exposure": rng.uniform(0.1, 1.0, n),
        "Area": rng.choice(list("ABC"), n),
        "VehBrand": rng.choice(["B1", "B2"], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Region": rng.choice(["R1", "R2"], n),
        "VehPower": rng.integers(4, 12, n),
        "VehAge": rng.integers(0, 40, n),
        "DrivAge": rng.integers(18, 100, n),
        "BonusMalus": rng.integers(50, 200, n),
        "Density": rng.integers(1, 5000, n),
    })
    rate = np.where(df.DrivAge < 25, 0.20, 0.05)
    df["ClaimNb"] = rng.poisson(rate * df.Exposure)
    return prepare_features(df)


def test_baseline_is_total_claims_over_total_exposure():
    b = NaiveBaseline().fit(np.array([1, 0, 2]), np.array([0.5, 0.5, 1.0]))
    assert b.rate_ == pytest.approx(1.5)
    assert b.predict(np.array([2.0])) == pytest.approx([3.0])


def test_prepare_features_caps_outliers(policies):
    assert policies.VehAge.max() <= 30
    assert policies.DrivAge.max() <= 90
    assert policies.BonusMalus.max() <= 150


def test_category_levels_survive_single_row_inference(policies):
    levels = get_category_levels(policies)
    row = pd.DataFrame([{c: levels[c][-1] for c in CATEGORICAL_FEATURES}])
    out = apply_category_levels(row, levels)
    assert list(out.Area.cat.categories) == levels["Area"]
    assert out.Area.cat.codes[0] == len(levels["Area"]) - 1


def test_unseen_category_becomes_nan_not_miscoded(policies):
    levels = get_category_levels(policies)
    row = pd.DataFrame([{**{c: levels[c][0] for c in CATEGORICAL_FEATURES}, "Area": "Z"}])
    assert apply_category_levels(row, levels).Area.isna().all()


def test_evaluate_prefers_the_true_rate(policies):
    y, e = policies.ClaimNb.values, policies.Exposure.values
    true = np.where(policies.DrivAge < 25, 0.20, 0.05) * e
    flat = np.full_like(e, y.sum() / e.sum()) * e
    assert evaluate_model("t", y, true, e)["poisson_deviance"] < evaluate_model("f", y, flat, e)["poisson_deviance"]


def test_glm_and_xgboost_predict_claim_counts_on_the_right_scale(policies):
    """Regression test for the base_score bug: XGBoost used to shrink every
    prediction ~20x, so total predicted claims were a tiny fraction of actual."""
    y, e = policies.ClaimNb.values, policies.Exposure.values
    glm = fit_poisson_glm(encode_for_glm(policies), y, e)
    xgb = fit_xgboost_poisson(encode_for_xgboost(policies), y, e)
    glm_total = (glm.predict(encode_for_glm(policies)) * e).sum()
    xgb_total = (xgb.predict(encode_for_xgboost(policies)) * e).sum()
    assert glm_total == pytest.approx(y.sum(), rel=0.1)
    assert xgb_total == pytest.approx(y.sum(), rel=0.1)


def test_xgboost_separates_high_risk_from_low_risk(policies):
    y, e = policies.ClaimNb.values, policies.Exposure.values
    xgb = fit_xgboost_poisson(encode_for_xgboost(policies), y, e)
    rate = xgb.predict(encode_for_xgboost(policies))
    young = policies.DrivAge < 25
    assert rate[young].mean() > 2 * rate[~young].mean()
