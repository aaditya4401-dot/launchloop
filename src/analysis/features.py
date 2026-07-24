"""
Phase 2 — shared feature engineering.

Both models are built on ONE feature table:
  - early-flight features: computed from just the first `WINDOW` seconds of each
    per-flight trace (the "predict the future" constraint for apogee).
  - target + label columns: final apogee (regression target) and is_weak_motor
    (the anomaly answer key), pulled from labels.parquet.

The early features are extracted with DuckDB reading the Parquet traces directly
(WHERE time <= WINDOW, grouped per flight) — the same data path as Phase 1, and
far faster than looping over 1,000 files in Python.

Feature reasoning (why each one):
  vz_max   — peak vertical velocity so far: proxy for kinetic energy -> apogee.
  az_max   — peak acceleration: thrust-to-weight ratio (motor strength / mass).
  az_mean  — average acceleration over the window: sustained push.
  alt_end  — altitude reached by the end of the window: head start on the climb.

Note: velocity is still rising throughout the first 2 s (motor burns to 3.9 s),
so "velocity at end of window" equals vz_max exactly — it was dropped as a
perfectly-collinear duplicate that would corrupt linear-regression coefficients.
"""

from __future__ import annotations

import duckdb
import polars as pl

WINDOW = 2.0  # seconds of early flight used as model input

# Observed early-flight features. Deliberately NO ground-truth knobs
# (thrust_scale / mass / is_weak_motor) — those would leak the answer.
FEATURE_COLS = ["vz_max", "az_max", "az_mean", "alt_end"]

TRACES_GLOB = "data/raw/flight_*.parquet"
LABELS_PATH = "data/labels.parquet"


def build_feature_table(window: float = WINDOW) -> pl.DataFrame:
    """Return one row per flight: early features + apogee target + weak label."""
    con = duckdb.connect()
    query = rf"""
        SELECT
            CAST(regexp_extract(filename, 'flight_(\d+)\.parquet', 1) AS INTEGER) AS flight_id,
            MAX(vertical_velocity)              AS vz_max,
            MAX(vertical_accel)                 AS az_max,
            AVG(vertical_accel)                 AS az_mean,
            arg_max(altitude_agl, time)         AS alt_end
        FROM read_parquet('{TRACES_GLOB}', filename = true)
        WHERE time <= {window}
        GROUP BY 1
    """
    # fetchnumpy() avoids needing pyarrow/pandas; polars builds straight from it.
    feats = pl.DataFrame(con.execute(query).fetchnumpy())
    con.close()

    labels = pl.read_parquet(LABELS_PATH).select(
        "flight_id", "apogee_agl", "is_weak_motor"
    )
    return feats.join(labels, on="flight_id", how="inner").sort("flight_id")


if __name__ == "__main__":
    df = build_feature_table()
    print(f"Feature table: {df.height} flights x {df.width} cols  (window = {WINDOW}s)")
    print("Feature columns:", FEATURE_COLS)
    print("\nHead:")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=1):
        print(df.head(5))
    print("\nTarget (apogee_agl) range:",
          round(df["apogee_agl"].min(), 1), "->", round(df["apogee_agl"].max(), 1), "m")
    print("Weak-motor flights:", int(df["is_weak_motor"].sum()),
          f"({100 * df['is_weak_motor'].mean():.1f}%)")
    print("\nAny nulls?:", df.null_count().sum_horizontal().item())
