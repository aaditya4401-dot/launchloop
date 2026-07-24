"""
Phase 3 — Step 1: the evaluate(config) -> metrics ORACLE.

This is the linchpin of the closed-loop designer. Every agent proposal is judged
here by a REAL RocketPy simulation — no metric is ever accepted unless the
simulator produced it.

A `Config` is a proposed rocket (which motor, how much nose ballast, launch-rail
angle, main-parachute size). `evaluate` flies it and returns `Metrics`:

  apogee_m              — how high it goes (checked against target & ceiling)
  stability_margin_cal  — static margin in calibers (hard safety band 1.5–2.5)
  rail_exit_velocity_ms — speed leaving the rail (hard safety, >= 20)
  landing_downrange_m   — nominal drift distance to the landing point
  landing_dispersion_m  — spread of the landing point under wind/build scatter
                          (a small manual Monte Carlo; the recovery objective)

The first four come from ONE deterministic flight. Dispersion needs randomness,
so it runs a small perturbed-flight ensemble (configurable; 0 skips it for speed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rocketpy import Flight

from src.simulate.rocket import (
    ELEVATION,
    HEADING,
    NOMINAL_MASS,
    RAIL_LENGTH,
    build_environment,
    build_motor_for_impulse,
    build_rocket,
)


@dataclass(frozen=True)
class Config:
    """A proposed rocket configuration — the agents' decision variables."""
    motor_total_impulse: float          # N·s, from the motor menu
    ballast_mass: float = 0.0           # kg of nose ballast
    rail_inclination: float = 85.0      # deg from horizontal (90 = straight up)
    parachute_cd_s: float = 10.0        # main-parachute drag area (cd * S)


@dataclass
class Metrics:
    """RocketPy-verified outcomes for a Config."""
    apogee_m: float
    stability_margin_cal: float
    rail_exit_velocity_ms: float
    landing_downrange_m: float
    landing_dispersion_m: float | None


def _fly(config: Config, wind_u: float, wind_v: float,
         mass: float = NOMINAL_MASS, impulse_factor: float = 1.0) -> Flight:
    """Build and fly one deterministic flight for a config (+ optional scatter)."""
    env = build_environment(wind_u, wind_v)
    motor = build_motor_for_impulse(config.motor_total_impulse * impulse_factor)
    rocket = build_rocket(
        mass, motor,
        ballast_mass=config.ballast_mass,
        main_cd_s=config.parachute_cd_s,
    )
    return Flight(
        rocket=rocket, environment=env,
        rail_length=RAIL_LENGTH,
        inclination=config.rail_inclination,
        heading=HEADING,
    )


def _landing_radius(flight: Flight) -> float:
    return math.hypot(float(flight.x_impact), float(flight.y_impact))


def evaluate(
    config: Config,
    wind_u: float,
    wind_v: float,
    dispersion_sims: int = 16,
    seed: int = 0,
) -> Metrics:
    """Fly `config` in the given wind and return RocketPy-verified metrics."""
    # --- one deterministic flight for the hard-constraint metrics ---
    flight = _fly(config, wind_u, wind_v)
    apogee = float(flight.apogee) - ELEVATION
    stability = float(flight.initial_stability_margin)
    rail_exit = float(flight.out_of_rail_velocity)
    downrange = _landing_radius(flight)

    # --- small manual Monte Carlo for landing dispersion (recovery objective) ---
    dispersion = None
    if dispersion_sims > 0:
        rng = np.random.default_rng(seed)
        radii = []
        for _ in range(dispersion_sims):
            wu = wind_u + rng.normal(0.0, 2.0)         # wind gust uncertainty
            wv = wind_v + rng.normal(0.0, 2.0)
            m = NOMINAL_MASS * rng.normal(1.0, 0.05)   # build mass scatter
            imp = rng.normal(1.0, 0.08)                # motor impulse scatter
            radii.append(_landing_radius(_fly(config, wu, wv, mass=m, impulse_factor=imp)))
        dispersion = float(np.std(radii, ddof=1))

    return Metrics(
        apogee_m=apogee,
        stability_margin_cal=stability,
        rail_exit_velocity_ms=rail_exit,
        landing_downrange_m=downrange,
        landing_dispersion_m=dispersion,
    )


if __name__ == "__main__":
    # Standalone smoke test: show that each knob moves the metrics as expected.
    # Mission wind for the test: a 6 m/s easterly.
    WIND_U, WIND_V = 6.0, 0.0

    trials = {
        "nominal (6000 N·s, no ballast)": Config(motor_total_impulse=6000),
        "+2 kg ballast":                  Config(motor_total_impulse=6000, ballast_mass=2.0),
        "smaller motor (3500 N·s)":       Config(motor_total_impulse=3500),
        "bigger motor (9000 N·s)":        Config(motor_total_impulse=9000),
        "bigger chute (cd·S 15)":         Config(motor_total_impulse=6000, parachute_cd_s=15.0),
        "less vertical (rail 75°)":       Config(motor_total_impulse=6000, rail_inclination=75.0),
    }

    print(f"Oracle test — wind = ({WIND_U}, {WIND_V}) m/s\n")
    hdr = f"{'config':32s} {'apogee':>8s} {'stab':>6s} {'railV':>6s} {'downR':>7s} {'disp':>6s}"
    print(hdr); print("-" * len(hdr))
    for name, cfg in trials.items():
        m = evaluate(cfg, WIND_U, WIND_V, dispersion_sims=12)
        print(f"{name:32s} {m.apogee_m:8.0f} {m.stability_margin_cal:6.2f} "
              f"{m.rail_exit_velocity_ms:6.1f} {m.landing_downrange_m:7.0f} "
              f"{m.landing_dispersion_m:6.0f}")
    print("\nunits: apogee m · stab cal · railV m/s · downR m · disp m")
