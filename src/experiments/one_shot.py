"""
One-shot (single-step) design baseline — the control arm for RQ1.

The question RQ1 asks is: does the propose -> verify -> correct LOOP add value
beyond a single informed proposal? This baseline isolates that. The brain gets
ONE oracle evaluation (of the mission's start config), the specialists and
orchestrator run exactly ONCE, and the orchestrator's single decision is
COMMITTED with no re-evaluation or iteration.

We then AUDIT the committed config with the oracle. That audit is the honesty
check: a one-shot agent can *assert* it fixed the design, but only the simulator
says whether the committed config actually satisfies the constraints. The audit
is a measurement, not part of the method's simulation budget (the agent consumed
exactly one oracle look before committing).

Contrast with the full loop (src/agents/designer.py), which re-simulates every
proposal and keeps correcting under the mission's iteration cap.
"""

from __future__ import annotations

from src.agents.brain import ROLES, Brain
from src.agents.constraints import check_constraints
from src.agents.mission import Mission
from src.agents.oracle import evaluate


def run_one_shot(mission: Mission, brain: Brain) -> dict:
    """Run the single-step baseline for one mission; return a result row."""
    start = mission.start_config

    # The one oracle look the agent is allowed before committing.
    metrics = evaluate(start, mission.wind_u, mission.wind_v,
                       dispersion_sims=mission.dispersion_sims)
    report = check_constraints(metrics, mission)

    # A strong single proposal. StubBrain exposes `cold_propose` (a compound
    # best-guess from the one look) so it isn't hobbled by its one-lever-per-turn
    # loop policy; LLM brains already propose compound changes, so we use their
    # orchestrator directly (run once, no loop).
    if hasattr(brain, "cold_propose"):
        decision = brain.cold_propose(mission, start, metrics, report)
    else:
        opinions = {r: brain.specialist_opinion(r, mission, start, metrics, report)
                    for r in ROLES}
        decision = brain.orchestrate(mission, start, metrics, report, opinions, history=[])

    # Commit to a single config from that one decision -- no correction loop.
    if decision.kind == "no_go":
        committed = start
        agent_verdict = "no_go"
    elif decision.kind == "go":
        committed = start
        agent_verdict = "go"
    else:  # propose -> commit to the proposed config as the final answer
        committed = decision.config
        agent_verdict = "go"  # the method "ships" this config

    # Audit the committed config against ground truth (not counted as a sim).
    audit_metrics = evaluate(committed, mission.wind_u, mission.wind_v,
                             dispersion_sims=mission.dispersion_sims)
    audit_report = check_constraints(audit_metrics, mission)

    return {
        "arm": "one_shot",
        "agent_verdict": agent_verdict,
        "success": bool(audit_report.success),
        # agent claimed a flyable design but the oracle disagrees:
        "hidden_violation": agent_verdict == "go" and not audit_report.success,
        "oracle_sims": 1,               # the single look before committing
        "final_config": committed,
        "audit_metrics": audit_metrics,
        "audit_report": audit_report,
    }
