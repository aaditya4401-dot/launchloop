"""
Tests for build_mission (src/agents/designer.py) — turns the designer CLI's
--target/--ceiling/--wind/--max-iterations flags into a Mission. Pure dataclass
logic, no RocketPy calls.
"""

from src.agents.designer import build_mission
from src.agents.mission import DEFAULT_MISSION


def test_no_overrides_returns_the_default_mission():
    assert build_mission() == DEFAULT_MISSION


def test_target_override_only_changes_target():
    mission = build_mission(target=4000.0)
    assert mission.target_apogee_m == 4000.0
    assert mission.ceiling_m == DEFAULT_MISSION.ceiling_m
    assert mission.wind_u == DEFAULT_MISSION.wind_u


def test_all_overrides_apply_together():
    mission = build_mission(target=4000.0, ceiling=4500.0, wind=3.0, max_iterations=5)
    assert mission.target_apogee_m == 4000.0
    assert mission.ceiling_m == 4500.0
    assert mission.wind_u == 3.0
    assert mission.max_iterations == 5
    # untouched fields still come from the default mission
    assert mission.stability_min_cal == DEFAULT_MISSION.stability_min_cal
    assert mission.start_config == DEFAULT_MISSION.start_config
