"""
Phase 1 — dispersion study (RocketPy's native Monte Carlo).

This is the SECOND of RocketPy's two data paths, and it is deliberately separate
from the manual flight loop (src/simulate/run.py):

  - The manual loop flies individual Flight objects to capture per-flight
    *time-series* (what Phase 2 needs). RocketPy's MonteCarlo can't give that.
  - This module uses RocketPy's built-in `MonteCarlo` class to answer a different
    question: for ONE nominal rocket with realistic manufacturing/weather scatter,
    what is the *range* of outcomes? -> mean/std apogee, stability, rail-exit
    velocity, and the landing-point spread (the safety footprint).

It is a demonstration of RocketPy's dispersion feature, not an input to Phase 2,
so it has its own `make dispersion` target and does not touch the make data chain.

Dispersion knobs (mirroring the loop's spirit): motor total impulse +/-8%,
airframe mass +/-5%, and wind scaled by a random factor, plus small launch-angle
scatter. Stochastic API verified against RocketPy 1.12.1 (a bare scalar argument
means "std dev, nominal from the base object"; a (nominal, std) tuple is explicit).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import typer
from rocketpy import (
    Flight,
    MonteCarlo,
    NoseCone,
    StochasticEnvironment,
    StochasticFlight,
    StochasticNoseCone,
    StochasticParachute,
    StochasticRocket,
    StochasticSolidMotor,
    StochasticTail,
    StochasticTrapezoidalFins,
    Tail,
    TrapezoidalFins,
)

from src.simulate.rocket import (
    ELEVATION,
    HEADING,
    INCLINATION,
    MOTOR_POSITION,
    NOMINAL_MASS,
    RAIL_LENGTH,
    build_environment,
    build_motor,
    build_rocket,
)

OUT_DIR = Path("data/dispersion")
NOMINAL_WIND_U = 5.0   # a steady 5 m/s easterly nominal wind
NOMINAL_WIND_V = 0.0

app = typer.Typer(add_completion=False, help="RocketPy native Monte Carlo dispersion study.")


def _build_stochastic():
    """Assemble the nominal flight and its stochastic (dispersed) wrappers."""
    env = build_environment(NOMINAL_WIND_U, NOMINAL_WIND_V)
    motor = build_motor(1.0)
    rocket = build_rocket(NOMINAL_MASS, motor)
    flight = Flight(
        rocket=rocket, environment=env,
        rail_length=RAIL_LENGTH, inclination=INCLINATION, heading=HEADING,
    )

    s_env = StochasticEnvironment(
        environment=env,
        wind_velocity_x_factor=(1, 0.3),   # +/-30% wind scatter
        wind_velocity_y_factor=(1, 0.3),
    )
    s_motor = StochasticSolidMotor(
        solid_motor=motor,
        total_impulse=(motor.total_impulse, 0.08 * motor.total_impulse),  # +/-8%
    )
    s_rocket = StochasticRocket(
        rocket=rocket,
        mass=(NOMINAL_MASS, 0.05 * NOMINAL_MASS),   # +/-5% airframe mass
    )
    s_rocket.add_motor(s_motor, position=(MOTOR_POSITION, 0))  # fixed position

    # StochasticRocket rebuilds the rocket from scratch: it does NOT inherit the
    # base rocket's aerosurfaces or parachutes. We must re-attach them, wrapped
    # in their Stochastic* types with no geometric dispersion (position std = 0).
    # Without this the recreated rocket has no fins (unstable) and no chutes
    # (ballistic impact) — verified against RocketPy 1.12.1.
    for surf in rocket.aerodynamic_surfaces:
        comp = surf.component
        pos = float(surf.position.z)
        if isinstance(comp, NoseCone):
            s_rocket.add_nose(StochasticNoseCone(nosecone=comp), position=(pos, 0))
        elif isinstance(comp, TrapezoidalFins):
            s_rocket.add_trapezoidal_fins(
                StochasticTrapezoidalFins(trapezoidal_fins=comp), position=(pos, 0)
            )
        elif isinstance(comp, Tail):
            s_rocket.add_tail(StochasticTail(tail=comp), position=(pos, 0))
    for chute in rocket.parachutes:
        s_rocket.add_parachute(StochasticParachute(parachute=chute))

    s_flight = StochasticFlight(
        flight=flight,
        inclination=(INCLINATION, 1.0),   # +/-1 deg launch-angle scatter
        heading=(HEADING, 3.0),           # +/-3 deg heading scatter
    )
    return s_env, s_rocket, s_flight


def _summarize(results: dict) -> pl.DataFrame:
    """Turn raw per-sim samples into a mean/std/min/max/p5/p95 summary table."""
    apogee_agl = np.asarray(results["apogee"]) - ELEVATION
    metrics = {
        "apogee_agl_m": apogee_agl,
        "apogee_time_s": np.asarray(results["apogee_time"]),
        "rail_exit_velocity_ms": np.asarray(results["out_of_rail_velocity"]),
        "stability_margin_cal": np.asarray(results["initial_stability_margin"]),
        "max_mach": np.asarray(results["max_mach_number"]),
        "impact_velocity_ms": np.asarray(results["impact_velocity"]),
    }
    rows = []
    for name, vals in metrics.items():
        rows.append({
            "metric": name,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p5": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
        })
    return pl.DataFrame(rows)


def _landing_summary(results: dict) -> dict:
    """Landing-point spread — the ground safety footprint."""
    x = np.asarray(results["x_impact"])
    y = np.asarray(results["y_impact"])
    radius = np.sqrt(x**2 + y**2)
    return {
        "mean_downrange_m": float(np.mean(radius)),
        "std_x_m": float(np.std(x, ddof=1)),
        "std_y_m": float(np.std(y, ddof=1)),
        "p95_radius_m": float(np.percentile(radius, 95)),
        "max_radius_m": float(np.max(radius)),
    }


@app.command()
def main(
    n: int = typer.Option(300, help="Number of dispersion simulations."),
    smoke: bool = typer.Option(False, help="Quick 10-sim smoke run."),
):
    """Run RocketPy's Monte Carlo dispersion and print/save a summary."""
    if smoke:
        n = 10
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s_env, s_rocket, s_flight = _build_stochastic()
    mc = MonteCarlo(
        filename=str(OUT_DIR / "mc"),
        environment=s_env, rocket=s_rocket, flight=s_flight,
    )
    # Discard the per-sim inputs log: RocketPy densifies the (constant) drag
    # Function to ~7.7M chars PER simulation when serializing inputs, so this
    # file balloons to gigabytes (4.3 GB for 300 sims). We derive our whole
    # summary from mc.results (the outputs), so the inputs log is pure waste.
    mc.input_file = os.devnull
    print(f"Running RocketPy MonteCarlo: {n} simulations "
          f"(nominal rocket + realistic scatter)...\n")
    mc.simulate(number_of_simulations=n, parallel=False)

    summary = _summarize(mc.results)
    landing = _landing_summary(mc.results)

    print("\n=== Dispersion summary (per-metric across", n, "flights) ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200,
                   float_precision=1):
        print(summary)

    print("\n=== Landing-point spread (ground safety footprint) ===")
    for k, v in landing.items():
        print(f"  {k:20s} {v:8.1f} m")

    # Persist the summary for the README.
    summary.write_parquet(OUT_DIR / "summary.parquet")
    print(f"\nSaved dispersion summary -> {OUT_DIR / 'summary.parquet'}")
    print(f"RocketPy raw per-sim outputs -> {OUT_DIR}/mc.outputs.txt "
          f"(inputs log discarded — see note in code)")


if __name__ == "__main__":
    app()
