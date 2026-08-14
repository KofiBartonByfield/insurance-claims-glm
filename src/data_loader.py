"""
data_loader.py

Downloads the French Motor Third-Party Liability datasets (freMTPL2) from
OpenML and merges the claim-frequency and claim-severity tables into a
single policy-level dataframe.

Source:
    freMTPL2freq — https://www.openml.org/d/41214 (one row per policy)
    freMTPL2sev  — https://www.openml.org/d/41215 (one row per claim)

No local data files are required — everything is fetched at runtime and
cached by scikit-learn in ~/scikit_learn_data on first run.
"""

import pandas as pd
from sklearn.datasets import fetch_openml


def load_raw_data():
    """Download the frequency and severity tables from OpenML."""
    freq = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    sev = fetch_openml(data_id=41215, as_frame=True, parser="auto").frame
    return freq, sev


def merge_frequency_severity(freq: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate claim amounts per policy and merge onto the frequency table.
    A small number of policies have more than one claim in the year —
    severity is summed per policy so each row stays one policy.
    """
    sev_agg = sev.groupby("IDpol", as_index=False)["ClaimAmount"].sum()
    df = freq.merge(sev_agg, on="IDpol", how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic sanity cleaning, consistent with the standard treatment of this
    dataset in actuarial literature (a handful of rows have data-entry
    errors rather than genuine extreme values).
    """
    df = df.copy()

    # Exposure is a fraction of a year and should not exceed 1.
    df["Exposure"] = df["Exposure"].clip(upper=1)

    # A small number of policies show implausibly high claim counts for a
    # single year — cap to keep the Poisson assumption reasonable.
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)

    # Cap extreme claim amounts (data errors, not genuine large losses) at
    # the 99.9th percentile.
    cap = df["ClaimAmount"].quantile(0.999)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=cap)

    return df.reset_index(drop=True)


def load_data() -> pd.DataFrame:
    """Full load: fetch, merge, clean. This is the only function main.py calls."""
    freq, sev = load_raw_data()
    df = merge_frequency_severity(freq, sev)
    return clean_data(df)
