"""
Tests for StubBrain (src/agents/brain.py) — the deterministic policy used to
build/test the LangGraph loop offline. All Metrics here are constructed directly
(no RocketPy calls) so we can hit exact decision boundaries, including the one
edge case where the policy has to admit defeat (no_go).
"""

from src.agents.brain import StubBrain
from src.agents.constraints import check_constraints
from src.agents.mission import DEFAULT_MISSION, MOTOR_BY_NAME
from src.agents.oracle import Config, Metrics

BRAIN = StubBrain()


def _metrics(apogee=3000.0, stability=2.0, rail_exit=30.0, dispersion=300.0):
    return Metrics(
        apogee_m=apogee, stability_margin_cal=stability,
        rail_exit_velocity_ms=rail_exit, landing_downrange_m=1000.0,
        landing_dispersion_m=dispersion,
    )


def _decide(config, metrics):
    report = check_constraints(metrics, DEFAULT_MISSION)
    return BRAIN.orchestrate(DEFAULT_MISSION, config, metrics, report, {}, [])


# --- orchestrator ------------------------------------------------------------
def test_go_when_all_hard_constraints_pass_and_on_target():
    config = Config(motor_total_impulse=MOTOR_BY_NAME["M1540"].total_impulse)
    decision = _decide(config, _metrics(apogee=3000.0))
    assert decision.kind == "go"


def test_downsizes_motor_when_apogee_exceeds_ceiling():
    config = Config(motor_total_impulse=MOTOR_BY_NAME["M2310"].total_impulse)
    decision = _decide(config, _metrics(apogee=5000.0))
    assert decision.kind == "propose"
    assert decision.config.motor_total_impulse < config.motor_total_impulse


def test_adds_ballast_when_smallest_motor_still_exceeds_ceiling():
    config = Config(motor_total_impulse=MOTOR_BY_NAME["L900"].total_impulse)
    decision = _decide(config, _metrics(apogee=5000.0))
    assert decision.kind == "propose"
    assert decision.config.ballast_mass > config.ballast_mass


def test_no_go_when_largest_motor_and_zero_ballast_cannot_fix_rail_exit():
    # On target and stable, but rail-exit too slow, on the biggest motor with no
    # ballast to shed -- the stub's fixed policy has no lever left to pull.
    config = Config(motor_total_impulse=MOTOR_BY_NAME["M2310"].total_impulse,
                    ballast_mass=0.0)
    decision = _decide(config, _metrics(apogee=3000.0, stability=2.0, rail_exit=15.0))
    assert decision.kind == "no_go"


# --- specialists (each holds ONE objective) ----------------------------------
def test_performance_dissatisfied_when_far_short_of_target():
    config = Config(motor_total_impulse=MOTOR_BY_NAME["L900"].total_impulse)
    metrics = _metrics(apogee=1000.0)
    report = check_constraints(metrics, DEFAULT_MISSION)
    op = BRAIN.specialist_opinion("performance", DEFAULT_MISSION, config, metrics, report)
    assert not op.satisfied


def test_safety_dissatisfied_when_ceiling_is_violated():
    config = Config(motor_total_impulse=MOTOR_BY_NAME["M2310"].total_impulse)
    metrics = _metrics(apogee=5000.0)
    report = check_constraints(metrics, DEFAULT_MISSION)
    op = BRAIN.specialist_opinion("safety", DEFAULT_MISSION, config, metrics, report)
    assert not op.satisfied


def test_recovery_dissatisfied_when_dispersion_is_large():
    config = Config(motor_total_impulse=MOTOR_BY_NAME["M1540"].total_impulse)
    metrics = _metrics(apogee=3000.0, dispersion=500.0)  # > 400 m threshold
    report = check_constraints(metrics, DEFAULT_MISSION)
    op = BRAIN.specialist_opinion("recovery", DEFAULT_MISSION, config, metrics, report)
    assert not op.satisfied


def test_specialists_can_disagree_on_the_same_design():
    # The load-bearing property of the multi-agent design: performance is happy
    # with a high apogee while safety is not, on the identical config/metrics.
    config = Config(motor_total_impulse=MOTOR_BY_NAME["M2310"].total_impulse)
    metrics = _metrics(apogee=5000.0)  # over target AND over the ceiling
    report = check_constraints(metrics, DEFAULT_MISSION)
    performance = BRAIN.specialist_opinion("performance", DEFAULT_MISSION, config, metrics, report)
    safety = BRAIN.specialist_opinion("safety", DEFAULT_MISSION, config, metrics, report)
    assert performance.satisfied and not safety.satisfied
