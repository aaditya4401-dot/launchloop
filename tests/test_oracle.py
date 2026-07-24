"""
Tests for the oracle (src/agents/oracle.py) — the ground-truth judge every agent
proposal is checked against. These assert the physics invariants the whole Phase 3
design depends on: each knob must move the metrics the way the agents assume it
does, or their reasoning ("add ballast to fix stability") would be nonsense.

dispersion_sims=0 everywhere except the two tests that specifically exercise
dispersion, to keep the suite fast (each real flight sim is ~0.3-0.5s).
"""

from src.agents.oracle import Config, evaluate

WIND_U, WIND_V = 6.0, 0.0


def test_ballast_lowers_apogee_and_raises_stability():
    # The core performance<->safety tension the multi-agent design relies on:
    # ballast trades altitude for stability margin.
    base = evaluate(Config(motor_total_impulse=6000.0), WIND_U, WIND_V, dispersion_sims=0)
    ballasted = evaluate(Config(motor_total_impulse=6000.0, ballast_mass=2.0),
                        WIND_U, WIND_V, dispersion_sims=0)

    assert ballasted.apogee_m < base.apogee_m
    assert ballasted.stability_margin_cal > base.stability_margin_cal


def test_bigger_motor_raises_apogee_and_rail_exit_velocity():
    small = evaluate(Config(motor_total_impulse=3500.0), WIND_U, WIND_V, dispersion_sims=0)
    big = evaluate(Config(motor_total_impulse=9000.0), WIND_U, WIND_V, dispersion_sims=0)

    assert big.apogee_m > small.apogee_m
    assert big.rail_exit_velocity_ms > small.rail_exit_velocity_ms


def test_parachute_size_changes_downrange_but_not_apogee():
    # The chute only deploys after apogee, so it must not affect how high the
    # rocket goes -- only how far it drifts on the way down.
    small_chute = evaluate(Config(motor_total_impulse=6000.0, parachute_cd_s=10.0),
                          WIND_U, WIND_V, dispersion_sims=0)
    big_chute = evaluate(Config(motor_total_impulse=6000.0, parachute_cd_s=15.0),
                        WIND_U, WIND_V, dispersion_sims=0)

    assert big_chute.apogee_m == small_chute.apogee_m
    assert big_chute.landing_downrange_m > small_chute.landing_downrange_m


def test_less_vertical_rail_reduces_apogee_and_increases_downrange():
    vertical = evaluate(Config(motor_total_impulse=6000.0, rail_inclination=85.0),
                       WIND_U, WIND_V, dispersion_sims=0)
    angled = evaluate(Config(motor_total_impulse=6000.0, rail_inclination=75.0),
                     WIND_U, WIND_V, dispersion_sims=0)

    assert angled.apogee_m < vertical.apogee_m
    assert angled.landing_downrange_m > vertical.landing_downrange_m


def test_zero_dispersion_sims_skips_the_monte_carlo():
    metrics = evaluate(Config(motor_total_impulse=6000.0), WIND_U, WIND_V, dispersion_sims=0)
    assert metrics.landing_dispersion_m is None


def test_positive_dispersion_sims_returns_a_positive_spread():
    metrics = evaluate(Config(motor_total_impulse=6000.0), WIND_U, WIND_V,
                       dispersion_sims=6, seed=1)
    assert metrics.landing_dispersion_m is not None
    assert metrics.landing_dispersion_m > 0
