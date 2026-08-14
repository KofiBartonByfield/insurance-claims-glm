"""
main.py

End-to-end pipeline: load data -> engineer features -> fit three
claim-frequency models -> evaluate -> save results and a lift chart.

Run with:
    python main.py

Requires an internet connection on first run (data is fetched from OpenML
and cached locally afterwards by scikit-learn).
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.features import (
    prepare_features,
    encode_for_glm,
    encode_for_xgboost,
    get_category_levels,
)
from src.models import NaiveBaseline, fit_poisson_glm, fit_xgboost_poisson
from src.evaluate import evaluate_model, lift_chart

OUTPUT_DIR = "outputs"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data from OpenML (freMTPL2freq + freMTPL2sev)...")
    df = load_data()
    df = prepare_features(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    y_train, y_test = train_df["ClaimNb"].values, test_df["ClaimNb"].values
    exp_train, exp_test = train_df["Exposure"].values, test_df["Exposure"].values

    # --- Naive baseline ---
    baseline = NaiveBaseline().fit(y_train, exp_train)
    baseline_pred = baseline.predict(exp_test)

    # --- Poisson GLM ---
    X_train_glm = encode_for_glm(train_df)
    X_test_glm = encode_for_glm(test_df).reindex(columns=X_train_glm.columns, fill_value=0)
    glm = fit_poisson_glm(X_train_glm, y_train, exp_train)
    glm_pred = glm.predict(X_test_glm) * exp_test

    # --- XGBoost (Poisson) ---
    X_train_xgb = encode_for_xgboost(train_df)
    X_test_xgb = encode_for_xgboost(test_df)
    xgb = fit_xgboost_poisson(X_train_xgb, y_train, exp_train)
    # base_margin was supplied at train time (log-exposure offset), so predict()
    # without it returns a RATE, not a count — scale by exposure to match the
    # GLM and baseline, which are both already expressed as counts. Comparing
    # an un-scaled rate against count-scale predictions would silently distort
    # the deviance comparison below.
    xgb_rate_pred = xgb.predict(X_test_xgb)
    xgb_pred = xgb_rate_pred * exp_test

    # --- Evaluate ---
    results = [
        evaluate_model("Naive baseline", y_test, baseline_pred, exp_test),
        evaluate_model("Poisson GLM", y_test, glm_pred, exp_test),
        evaluate_model("XGBoost (Poisson)", y_test, xgb_pred, exp_test),
    ]
    results_df = pd.DataFrame(results)
    print("\nResults:\n", results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)

    # --- Lift chart ---
    lift_chart(
        y_test,
        {"Naive baseline": baseline_pred, "Poisson GLM": glm_pred, "XGBoost": xgb_pred},
        exp_test,
        save_path=os.path.join(OUTPUT_DIR, "lift_chart.png"),
    )
    print(f"\nSaved results.csv and lift_chart.png to {OUTPUT_DIR}/")

    # --- Persist the XGBoost model for serving ---
    # Feature columns and categorical levels are saved alongside the model
    # because the serving layer needs to reproduce training-time encoding
    # exactly — see src/features.py:apply_category_levels for why this matters.
    artifacts = {
        "model": xgb,
        "feature_columns": list(X_train_xgb.columns),
        "categorical_levels": get_category_levels(train_df),
    }
    model_path = os.path.join(OUTPUT_DIR, "xgb_poisson_model.joblib")
    joblib.dump(artifacts, model_path)
    print(f"Saved model artifacts to {model_path}")


if __name__ == "__main__":
    main()
