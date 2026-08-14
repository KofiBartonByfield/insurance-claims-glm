"""
app.py

Lightweight inference API for the claim-frequency model. Loads the
artifacts saved by main.py (model, expected feature columns, frozen
categorical levels) and serves predictions for one policy at a time.

Run locally with:
    uvicorn app:app --reload

Requires outputs/xgb_poisson_model.joblib to exist — run `python main.py`
first to train and save it.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from src.features import prepare_features, encode_for_xgboost, apply_category_levels

ARTIFACT_PATH = "outputs/xgb_poisson_model.joblib"
API_KEY = os.environ.get("API_KEY")  # set at deploy time; see README

app = FastAPI(title="Claim Frequency Prediction API")

try:
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts["model"]
    expected_columns = artifacts["feature_columns"]
    categorical_levels = artifacts["categorical_levels"]
except FileNotFoundError:
    model = None
    expected_columns = None
    categorical_levels = None


class PolicyHolderInput(BaseModel):
    Exposure: float
    Area: str
    VehPower: int
    VehAge: int
    DrivAge: int
    BonusMalus: int
    VehBrand: str
    VehGas: str
    Density: float
    Region: str


@app.get("/")
def health_check():
    return {"status": "ok" if model is not None else "model artifact not found"}


@app.post("/predict")
def predict(payload: PolicyHolderInput, x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded — run `python main.py` to produce {ARTIFACT_PATH}",
        )

    raw_df = pd.DataFrame([payload.model_dump()])

    processed_df = prepare_features(raw_df)
    # Freeze categorical levels to match training. Without this, a single-row
    # request gets its own one-off category encoding that won't line up with
    # what the model was trained on — see src/features.py for details.
    processed_df = apply_category_levels(processed_df, categorical_levels)

    encoded_df = encode_for_xgboost(processed_df).reindex(columns=expected_columns)

    # base_margin was used at train time as a log-exposure offset, so a plain
    # predict() call here returns a RATE, not a count — multiply by exposure
    # to get the expected number of claims, matching how main.py evaluates
    # this same model.
    predicted_rate = float(model.predict(encoded_df)[0])
    expected_claims = predicted_rate * payload.Exposure

    return {
        "predicted_claim_rate": predicted_rate,
        "expected_claims": expected_claims,
    }
