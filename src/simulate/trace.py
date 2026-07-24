"""
Phase 1 — Step 2: pull a flight's time-series out of a RocketPy Flight object
and save it as a Parquet file.

A RocketPy Flight exposes its results as callable `Function` objects (altitude,
vz, az) plus the solver's `time` grid. We sample those onto the time grid to get
a plain table: one row per timestamp. That table is what Phase 2's models read.

Columns (per PLAN.md):
    time                 seconds since launch
    altitude_agl         height above ground level (m)
    vertical_velocity    vertical speed (m/s)
    vertical_accel       vertical acceleration (m/s^2)
"""

from pathlib import Path

import numpy as np
import polars as pl


def flight_to_dataframe(flight) -> pl.DataFrame:
    """Sample a RocketPy Flight's vertical time-series onto its solver time grid."""
    t = np.asarray(flight.time, dtype=float)
    return pl.DataFrame(
        {
            "time": t,
            "altitude_agl": flight.altitude(t).astype(float),
            "vertical_velocity": flight.vz(t).astype(float),
            "vertical_accel": flight.az(t).astype(float),
        }
    )


def save_trace(flight, path: str | Path) -> Path:
    """Write a flight's time-series to a Parquet file, returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = flight_to_dataframe(flight)
    df.write_parquet(path)
    return path


if __name__ == "__main__":
    # Throwaway harness: build the one flight and save a single sample trace,
    # so we can inspect the real columns before scaling to the full loop.
    from src.simulate.single_flight import flight

    out = save_trace(flight, "data/raw/flight_sample.parquet")
    df = flight_to_dataframe(flight)
    print(f"\nSaved {out}  ({out.stat().st_size:,} bytes, {df.height:,} rows)")
    print("\nSchema:")
    for col, dtype in df.schema.items():
        print(f"  {col:20s} {dtype}")
    print("\nFirst 5 rows:")
    print(df.head(5))
    print("Row at apogee-ish (max altitude):")
    print(df.sort("altitude_agl", descending=True).head(1))
