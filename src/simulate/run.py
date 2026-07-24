"""
Phase 1 — Step 3: the Monte Carlo flight loop.

Flies the parametric rocket many times, jittering four knobs per flight, saving
each flight's time-series as data/raw/flight_XXXX.parquet, and accumulating one
label row per flight into data/labels.parquet.

Per-flight randomization (implemented exactly as specified):
  - Wind:   heading theta ~ U(0, 2*pi), magnitude m ~ U(0, 20) m/s, then
            wind_u = m*cos(theta), wind_v = m*sin(theta).  Drawing magnitude +
            direction (not u, v independently) keeps total wind bounded and
            direction uniform instead of clumping diagonally.
  - Mass:   Normal(nominal, 5% of nominal).
  - Motor:  thrust curve scaled by Normal(1.0, 0.08)  (+/-8%).
  - Weak motor (~10% of flights): thrust scaled to U(0.55, 0.70) instead -> low
            total impulse. Labeled is_weak_motor=True; these are the Phase 2
            "answer key".

Reproducibility: each flight draws from its own RNG seeded base_seed + flight_id,
and draws happen in a fixed order, so any single flight reproduces exactly
without rerunning the batch.

Run:  uv run python -m src.simulate.run            # full 1000-flight batch
      uv run python -m src.simulate.run --smoke    # 5-flight smoke test
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import typer

from src.simulate.rocket import ELEVATION, NOMINAL_MASS, build_flight
from src.simulate.trace import save_trace

RAW_DIR = Path("data/raw")
LABELS_PATH = Path("data/labels.parquet")

# Weak-motor thrust band. Deliberately LOW (55-70%) for now so Phase 2 anomaly
# detection has an easy, obvious signal to catch first. Tighten to ~0.75-0.85
# later to make the detection problem less trivial.
WEAK_THRUST_LOW = 0.55
WEAK_THRUST_HIGH = 0.70
WEAK_FRACTION = 0.10

app = typer.Typer(add_completion=False, help="Phase 1 Monte Carlo flight loop.")


def draw_params(
    flight_id: int,
    base_seed: int,
    force_weak: bool = False,
    force_thrust_scale: float | None = None,
) -> dict:
    """Draw one flight's knob values reproducibly from base_seed + flight_id.

    Draws always happen in the same order so a flight's underlying values are
    fixed regardless of forcing; forcing only *selects* among values already
    drawn (or overrides the thrust scale outright for the smoke test).
    """
    rng = np.random.default_rng(base_seed + flight_id)

    # --- wind: magnitude + direction (never u, v independently) ---
    theta = rng.uniform(0.0, 2.0 * math.pi)
    magnitude = rng.uniform(0.0, 20.0)
    wind_u = magnitude * math.cos(theta)
    wind_v = magnitude * math.sin(theta)

    # --- airframe mass ---
    mass = float(rng.normal(NOMINAL_MASS, 0.05 * NOMINAL_MASS))

    # --- motor: draw both branches up front to keep the RNG order fixed ---
    weak_roll = rng.random()
    weak_scale = float(rng.uniform(WEAK_THRUST_LOW, WEAK_THRUST_HIGH))
    normal_scale = float(rng.normal(1.0, 0.08))

    is_weak = force_weak or (weak_roll < WEAK_FRACTION)
    thrust_scale = weak_scale if is_weak else normal_scale
    if force_thrust_scale is not None:
        thrust_scale = force_thrust_scale

    return {
        "flight_id": flight_id,
        "seed": base_seed + flight_id,
        "is_weak_motor": is_weak,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "wind_magnitude": magnitude,
        "wind_heading_rad": theta,
        "mass": mass,
        "thrust_scale": thrust_scale,
    }


def simulate_one(params: dict) -> dict:
    """Run one flight, save its trace Parquet, and return a label/summary row."""
    flight = build_flight(
        wind_u=params["wind_u"],
        wind_v=params["wind_v"],
        mass=params["mass"],
        thrust_scale=params["thrust_scale"],
    )

    trace_path = RAW_DIR / f"flight_{params['flight_id']:04d}.parquet"
    save_trace(flight, trace_path)

    # Outcome columns recorded alongside the input knobs.
    row = dict(params)
    row.update(
        {
            "apogee_agl": float(flight.apogee) - ELEVATION,
            "max_speed": float(flight.max_speed),
            "out_of_rail_velocity": float(flight.out_of_rail_velocity),
            "apogee_time": float(flight.apogee_time),
            "n_rows": int(np.asarray(flight.time).size),
            "trace_file": trace_path.name,
        }
    )
    return row


def _run_batch(specs: list[dict], base_seed: int) -> pl.DataFrame:
    """Simulate a list of param dicts, printing per-flight progress."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.perf_counter()
    for params in specs:
        t0 = time.perf_counter()
        row = simulate_one(params)
        row["sim_seconds"] = time.perf_counter() - t0
        rows.append(row)
        tag = "WEAK" if row["is_weak_motor"] else "ok  "
        print(
            f"  flight {row['flight_id']:04d} [{tag}] "
            f"apogee={row['apogee_agl']:7.1f} m  "
            f"vmax={row['max_speed']:6.1f} m/s  "
            f"rail_exit={row['out_of_rail_velocity']:5.1f} m/s  "
            f"thrust x{row['thrust_scale']:.2f}  "
            f"({row['sim_seconds']:.2f}s)"
        )
    elapsed = time.perf_counter() - t_start
    print(f"\n  {len(rows)} flights in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.2f}s/flight)")
    return pl.DataFrame(rows)


@app.command()
def main(
    n: int = typer.Option(1000, help="Number of flights in the full batch."),
    seed: int = typer.Option(42, help="Base seed; flight i uses seed + i."),
    smoke: bool = typer.Option(False, help="Run a 5-flight smoke test instead."),
):
    """Fly the rocket `n` times and write traces + labels."""
    # Clear previous trace files so the batch is a clean rebuild.
    for old in RAW_DIR.glob("flight_*.parquet"):
        old.unlink()

    if smoke:
        print("SMOKE TEST — 5 flights; flight 0003 forced to a very weak motor "
              f"(thrust x{WEAK_THRUST_LOW:.2f}) to expose the barely-clears-rail case.\n")
        specs = []
        for fid in range(1, 6):
            if fid == 3:
                specs.append(draw_params(fid, seed, force_weak=True,
                                         force_thrust_scale=WEAK_THRUST_LOW))
            else:
                specs.append(draw_params(fid, seed))
    else:
        print(f"FULL BATCH — {n} flights (base seed {seed}).\n")
        specs = [draw_params(fid, seed) for fid in range(1, n + 1)]

    labels = _run_batch(specs, seed)

    if not smoke:
        LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
        labels.write_parquet(LABELS_PATH)
        n_weak = int(labels["is_weak_motor"].sum())
        print(f"\n  wrote {LABELS_PATH}  ({labels.height} rows, "
              f"{n_weak} weak = {100 * n_weak / labels.height:.1f}%)")
    else:
        # Smoke test: show the label table but don't overwrite the real labels file.
        print("\nSmoke-test label rows:")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(labels.select(
                "flight_id", "is_weak_motor", "wind_magnitude", "mass",
                "thrust_scale", "apogee_agl", "max_speed", "out_of_rail_velocity",
            ))


if __name__ == "__main__":
    app()
