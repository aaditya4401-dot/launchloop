"""
Tests for the mission/motor-menu definitions (src/agents/mission.py).
"""

from src.agents.mission import DEFAULT_MISSION, MOTOR_BY_NAME, MOTOR_MENU
from src.agents.oracle import evaluate
from src.simulate.rocket import BURN_TIME


def test_motor_menu_is_sorted_small_to_large():
    impulses = [m.total_impulse for m in MOTOR_MENU]
    assert impulses == sorted(impulses)


def test_motor_avg_thrust_matches_impulse_over_burn_time():
    for m in MOTOR_MENU:
        assert m.avg_thrust == m.total_impulse / BURN_TIME


def test_motor_names_are_unique_and_lookup_is_consistent():
    names = [m.name for m in MOTOR_MENU]
    assert len(names) == len(set(names))
    for m in MOTOR_MENU:
        assert MOTOR_BY_NAME[m.name] is m


def test_default_mission_starts_on_the_biggest_motor():
    biggest = max(MOTOR_MENU, key=lambda m: m.total_impulse)
    assert DEFAULT_MISSION.start_config.motor_total_impulse == biggest.total_impulse


def test_default_mission_start_config_violates_the_ceiling_by_design():
    # Documented design choice in mission.py: the starting motor must overshoot
    # the ceiling so the agents are forced to actually correct the design rather
    # than pass on the first iteration. This is a regression guard for that intent.
    metrics = evaluate(DEFAULT_MISSION.start_config, DEFAULT_MISSION.wind_u,
                       DEFAULT_MISSION.wind_v, dispersion_sims=0)
    assert metrics.apogee_m > DEFAULT_MISSION.ceiling_m
