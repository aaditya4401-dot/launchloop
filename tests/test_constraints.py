"""
Tests for check_constraints (src/agents/constraints.py) — the deterministic
bookkeeping that turns raw oracle Metrics into pass/fail. No RocketPy calls here:
Metrics are constructed directly so we can hit exact boundary conditions.
"""

import pytest

from src.agents.constraints import check_constraints
from src.agents.mission import DEFAULT_MISSION
from src.agents.oracle import Metrics


def _metrics(apogee=3000.0, stability=2.0, rail_exit=30.0, dispersion=300.0):
    return Metrics(
        apogee_m=apogee, stability_margin_cal=stability,
        rail_exit_velocity_ms=rail_exit, landing_downrange_m=1000.0,
        landing_dispersion_m=dispersion,
    )


def test_all_hard_constraints_pass_within_the_safe_band():
    report = check_constraints(_metrics(apogee=3000.0), DEFAULT_MISSION)
    assert report.hard_ok
    assert report.violations == []


def test_ceiling_violation_is_flagged():
    report = check_constraints(_metrics(apogee=3500.0), DEFAULT_MISSION)  # ceiling is 3200
    assert not report.hard_ok
    assert any("ceiling" in v for v in report.violations)


def test_stability_outside_band_is_flagged():
    report = check_constraints(_metrics(stability=3.0), DEFAULT_MISSION)  # max is 2.5
    assert not report.hard_ok
    assert any("stability" in v for v in report.violations)


def test_rail_exit_below_minimum_is_flagged():
    report = check_constraints(_metrics(rail_exit=10.0), DEFAULT_MISSION)  # min is 20
    assert not report.hard_ok
    assert any("rail-exit" in v for v in report.violations)


def test_apogee_gap_is_signed_distance_from_target():
    report = check_constraints(_metrics(apogee=3050.0), DEFAULT_MISSION)  # target 3000
    assert report.apogee_gap_m == pytest.approx(50.0)


def test_success_requires_both_hard_ok_and_on_target():
    # Hard constraints pass (1000 m violates nothing) but it's nowhere near target.
    off_target = check_constraints(_metrics(apogee=1000.0), DEFAULT_MISSION)
    assert off_target.hard_ok
    assert not off_target.on_target
    assert not off_target.success

    on_target = check_constraints(_metrics(apogee=3000.0), DEFAULT_MISSION)
    assert on_target.success
