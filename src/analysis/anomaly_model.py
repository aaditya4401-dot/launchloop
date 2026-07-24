"""
Phase 2 — Model 2: anomaly detection (unsupervised).

Goal: flag "bad" flights WITHOUT training on the answer. We fit an
IsolationForest — an unsupervised outlier detector that isolates points which are
easy to separate from the crowd — on observed flight metrics, then use the
Phase 1 weak-motor labels ONLY to score it (precision / recall).

Fair-features rule: we use observed OUTCOMES a launch team could measure from
telemetry — apogee, max speed, rail-exit velocity, time-to-apogee — and NOT the
ground-truth knobs (thrust_scale / is_weak_motor), which would leak the answer.

A weak motor shows up as low-and-slow (low apogee, low speed, early apogee), so it
tends to sit in the outlier region. But because the weak/normal apogee ranges
OVERLAP (Phase 1), detection is imperfect — which is what makes precision/recall
worth reporting.

One-sided refinement: IsolationForest flags outliers in BOTH tails, but a sabotaged
motor can only make a flight WORSE. So we keep only anomalies on the underperformance
side (below-median apogee). This drops the high-performer false alarms — precision
0.71 -> 0.87 — with no loss of recall (every weak motor is an underperformer).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

NOTEBOOKS = Path("notebooks")
LABELS_PATH = "data/labels.parquet"
RANDOM_STATE = 42

# Observed-outcome features (no ground-truth knobs).
ANOMALY_FEATURES = ["apogee_agl", "max_speed", "out_of_rail_velocity", "apogee_time"]

# We expect ~10% bad flights, so tell IsolationForest to flag ~10% as outliers.
CONTAMINATION = 0.10


def detect():
    df = pl.read_parquet(LABELS_PATH).sort("flight_id")
    X = df.select(ANOMALY_FEATURES).to_numpy()
    y_true = df["is_weak_motor"].to_numpy().astype(int)  # answer key (1 = weak)

    Xs = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    raw = iso.fit_predict(Xs)              # -1 = anomaly, +1 = normal
    two_sided = (raw == -1).astype(int)    # outliers in BOTH tails
    scores = -iso.score_samples(Xs)        # higher = more anomalous

    # One-sided: keep only underperformance-side anomalies (below-median apogee),
    # since a sabotaged motor can only make a flight worse.
    apogee = df["apogee_agl"].to_numpy()
    median_apogee = float(np.median(apogee))
    flagged = (two_sided & (apogee < median_apogee)).astype(int)

    precision = precision_score(y_true, flagged)
    recall = recall_score(y_true, flagged)
    f1 = f1_score(y_true, flagged)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged).ravel()

    print("Anomaly detection — IsolationForest on observed flight metrics")
    print(f"Features: {ANOMALY_FEATURES}")
    print(f"Flights: {len(y_true)} | weak (truth): {y_true.sum()}")
    print(f"IsolationForest flagged {two_sided.sum()} (both tails, contamination="
          f"{CONTAMINATION}); one-sided filter (apogee < {median_apogee:.0f} m) "
          f"keeps {flagged.sum()}.\n")
    print(f"  Precision : {precision:.3f}  (of flagged flights, share truly weak)")
    print(f"  Recall    : {recall:.3f}  (of weak flights, share we caught)")
    print(f"  F1        : {f1:.3f}\n")
    print("  Confusion matrix (rows=truth, cols=flagged):")
    print(f"                 flagged=no   flagged=yes")
    print(f"    normal   {tn:>10d}   {fp:>11d}")
    print(f"    weak     {fn:>10d}   {tp:>11d}")
    print(f"\n  Caught {tp}/{tp + fn} weak motors; {fp} false alarms; {fn} missed.")

    _plot_scatter(df, y_true, flagged, median_apogee)
    _plot_score_hist(scores, y_true)
    return {"precision": precision, "recall": recall, "f1": f1}


def _plot_scatter(df, y_true, flagged, median_apogee):
    NOTEBOOKS.mkdir(exist_ok=True)
    apogee = df["apogee_agl"].to_numpy()
    speed = df["max_speed"].to_numpy()
    weak = y_true.astype(bool)
    flag = flagged.astype(bool)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    # base colour = truth (normal blue / weak red)
    ax.scatter(apogee[~weak], speed[~weak], s=16, alpha=0.5,
               color="#4C78A8", label="normal (truth)")
    ax.scatter(apogee[weak], speed[weak], s=28, alpha=0.8,
               color="#E45756", label="weak motor (truth)")
    # black ring = flagged as anomaly by the model
    ax.scatter(apogee[flag], speed[flag], s=90, facecolors="none",
               edgecolors="black", linewidths=1.1, label="flagged anomaly")
    ax.axvline(median_apogee, color="grey", ls=":", lw=1,
               label=f"one-sided cutoff ({median_apogee:.0f} m)")
    ax.set_xlabel("Apogee (m)")
    ax.set_ylabel("Max speed (m/s)")
    ax.set_title("Anomaly detection: flagged flights vs weak-motor truth")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out = NOTEBOOKS / "anomaly_scatter.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved chart -> {out}")


def _plot_score_hist(scores, y_true):
    weak = y_true.astype(bool)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(scores[~weak], bins=bins, alpha=0.7, color="#4C78A8", label="normal")
    ax.hist(scores[weak], bins=bins, alpha=0.8, color="#E45756", label="weak motor")
    ax.set_xlabel("Anomaly score (higher = more anomalous)")
    ax.set_ylabel("Number of flights")
    ax.set_title("Anomaly score separates weak motors from normal flights")
    ax.legend()
    fig.tight_layout()
    out = NOTEBOOKS / "anomaly_score_hist.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved chart -> {out}")


if __name__ == "__main__":
    detect()
