"""
features.py

Feature engineering for the freMTPL2 claim-frequency model.
"""

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cap long-tailed numeric fields and set categorical dtypes."""
    df = df.copy()

    df["VehAge"] = df["VehAge"].clip(upper=30)
    df["DrivAge"] = df["DrivAge"].clip(upper=90)
    df["BonusMalus"] = df["BonusMalus"].clip(upper=150)
    df["Density"] = np.log1p(df["Density"])

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    return df


def encode_for_glm(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals for the linear GLM."""
    return pd.get_dummies(df[ALL_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=True)


def encode_for_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost takes category dtype directly (enable_categorical=True in the model)."""
    return df[ALL_FEATURES]


def get_category_levels(df: pd.DataFrame) -> dict:
    """
    Capture the categorical levels seen in a (post prepare_features) dataframe,
    so they can be frozen and reused at inference time.
    """
    return {col: list(df[col].cat.categories) for col in CATEGORICAL_FEATURES}


def apply_category_levels(df: pd.DataFrame, levels: dict) -> pd.DataFrame:
    """
    Force categorical columns onto a fixed, pre-recorded set of levels.

    This matters most for single-row inference requests: without it, pandas
    builds each column's category dtype from whatever values are present in
    that one row, so a value like Area="B" could silently get assigned a
    different internal code than it had during training, corrupting the
    prediction with no error raised. Any category not seen during training
    becomes NaN here rather than being miscoded.
    """
    df = df.copy()
    for col, cats in levels.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df
