"""
evaluate.py

Evaluation metrics and a lift chart, following standard actuarial practice
for comparing frequency models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error


def evaluate_model(name, y_true, y_pred, exposure):
    """y_true / y_pred are both claim COUNTS (not rates), as required for Poisson deviance."""
    y_pred = np.clip(y_pred, 1e-6, None)
    deviance = mean_poisson_deviance(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {"model": name, "poisson_deviance": deviance, "mae": mae}


def lift_chart(y_true, preds_dict, exposure, n_bins=10, save_path=None):
    """
    Sorts policies into deciles of predicted risk and compares actual vs
    predicted claim frequency per decile — the standard actuarial check for
    whether a model actually separates risk, not just gets the average right.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, y_pred in preds_dict.items():
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "exposure": exposure})
        df["decile"] = pd.qcut(df["y_pred"], n_bins, labels=False, duplicates="drop")
        grouped = df.groupby("decile").apply(lambda g: g["y_true"].sum() / g["exposure"].sum())
        ax.plot(grouped.index, grouped.values, marker="o", label=name)

    ax.set_xlabel("Predicted risk decile (low to high)")
    ax.set_ylabel("Observed claim frequency")
    ax.set_title("Lift chart: observed frequency by predicted-risk decile")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
