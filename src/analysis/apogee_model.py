"""
Phase 2 — Model 1: apogee prediction (regression).

Predict each flight's FINAL apogee using only features from the first 2 seconds
of flight. We compare three things on a held-out test set:

  1. Baseline  — always guess the mean apogee (what you'd do with no model).
  2. Linear regression — interpretable; reads off which features matter.
  3. Random forest — captures nonlinearity; usually the lower error.

Metrics:
  MAE  — mean absolute error, in metres (typical miss).
  RMSE — root mean squared error, in metres (punishes big misses).
  R^2  — fraction of apogee variance explained (1.0 = perfect, 0 = no better
         than the mean baseline).

Charts saved to notebooks/ for the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display; render straight to files
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.features import FEATURE_COLS, WINDOW, build_feature_table

NOTEBOOKS = Path("notebooks")
RANDOM_STATE = 42


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate():
    df = build_feature_table()
    X = df.select(FEATURE_COLS).to_numpy()
    y = df["apogee_agl"].to_numpy()
    weak = df["is_weak_motor"].to_numpy()

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, weak, test_size=0.2, random_state=RANDOM_STATE
    )

    # 1) Baseline: always predict the training mean.
    baseline_pred = np.full_like(y_te, y_tr.mean())

    # 2) Linear regression (standardized features for readable coefficients).
    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(X_tr, y_tr)
    linear_pred = linear.predict(X_te)

    # 3) Random forest.
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)

    results = {
        "Baseline (mean)": _metrics(y_te, baseline_pred),
        "Linear regression": _metrics(y_te, linear_pred),
        "Random forest": _metrics(y_te, rf_pred),
    }

    # ---- report ----
    print(f"Apogee prediction — first {WINDOW:.0f}s features -> final apogee")
    print(f"Train/test: {len(y_tr)}/{len(y_te)} flights | features: {FEATURE_COLS}\n")
    print(f"{'model':20s} {'MAE (m)':>10s} {'RMSE (m)':>10s} {'R^2':>8s}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:20s} {m['MAE']:10.1f} {m['RMSE']:10.1f} {m['R2']:8.3f}")

    preds = {"Linear regression": linear_pred, "Random forest": rf_pred}
    best = min(preds, key=lambda n: results[n]["MAE"])
    improve = 100 * (1 - results[best]["MAE"] / results["Baseline (mean)"]["MAE"])
    print(f"\nBest model: {best} — MAE {results[best]['MAE']:.1f} m, "
          f"{improve:.1f}% lower error than guessing the mean "
          f"({results['Baseline (mean)']['MAE']:.0f} m).")

    # Linear coefficients (standardized => comparable feature weights).
    coefs = linear.named_steps["linearregression"].coef_
    print("\nLinear feature weights (standardized, larger |·| = more influence):")
    for name, c in sorted(zip(FEATURE_COLS, coefs), key=lambda t: -abs(t[1])):
        print(f"  {name:10s} {c:+10.1f} m per std")

    _plot_pred_vs_actual(y_te, preds[best], w_te, results[best], best)
    _plot_feature_importance(rf)
    return results


def _plot_pred_vs_actual(y_true, y_pred, weak, metrics, model_name):
    NOTEBOOKS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true[~weak], y_pred[~weak], s=18, alpha=0.6,
               label="normal", color="#4C78A8")
    ax.scatter(y_true[weak], y_pred[weak], s=28, alpha=0.8,
               label="weak motor", color="#E45756")
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    ax.set_xlabel("Actual apogee (m)")
    ax.set_ylabel("Predicted apogee (m)")
    ax.set_title(f"Apogee: predicted vs actual ({model_name})\n"
                 f"MAE {metrics['MAE']:.0f} m · RMSE {metrics['RMSE']:.0f} m · "
                 f"R² {metrics['R2']:.3f}")
    ax.legend()
    fig.tight_layout()
    out = NOTEBOOKS / "apogee_pred_vs_actual.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved chart -> {out}")


def _plot_feature_importance(rf):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    order = np.argsort(rf.feature_importances_)
    ax.barh([FEATURE_COLS[i] for i in order],
            rf.feature_importances_[order], color="#4C78A8")
    ax.set_xlabel("Random-forest feature importance")
    ax.set_title("Which early-flight features drive the apogee prediction")
    fig.tight_layout()
    out = NOTEBOOKS / "apogee_feature_importance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved chart -> {out}")


if __name__ == "__main__":
    train_and_evaluate()
