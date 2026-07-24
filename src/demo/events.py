"""
Phase 3 demo — event layer.

Runs the ACTUAL LangGraph design loop (build_graph from designer.py) and
translates its per-node stream into flat, JSON-serializable event dicts. This one
format is used by everything:

  - the Streamlit app renders these dicts live (streaming) AND on replay,
  - the recorder dumps a list of them to JSON for the no-key cached demo.

No agent logic or oracle is reimplemented here — we only run the graph and shape
its output.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

from src.agents.brain import ROLES, Brain
from src.agents.designer import build_graph
from src.agents.mission import MOTOR_MENU, Mission


def _motor_name(impulse: float) -> str:
    return next((m.name for m in MOTOR_MENU if m.total_impulse == impulse),
                f"{impulse:.0f}N·s")


def mission_to_dict(m: Mission) -> dict:
    return {
        "target_apogee_m": m.target_apogee_m,
        "ceiling_m": m.ceiling_m,
        "stability_min_cal": m.stability_min_cal,
        "stability_max_cal": m.stability_max_cal,
        "rail_exit_min_ms": m.rail_exit_min_ms,
        "target_tolerance_m": m.target_tolerance_m,
        "wind_u": m.wind_u,
        "wind_v": m.wind_v,
        "max_iterations": m.max_iterations,
    }


def _config_to_dict(c) -> dict:
    return {
        "motor": _motor_name(c.motor_total_impulse),
        "impulse": c.motor_total_impulse,
        "ballast_kg": c.ballast_mass,
        "rail_deg": c.rail_inclination,
        "chute_cd_s": c.parachute_cd_s,
    }


def _metrics_to_dict(mt) -> dict:
    return {
        "apogee_m": mt.apogee_m,
        "stability_cal": mt.stability_margin_cal,
        "rail_exit_ms": mt.rail_exit_velocity_ms,
        "dispersion_m": mt.landing_dispersion_m,
    }


def _config_change(old, new) -> str:
    """One-line human summary of what the orchestrator changed."""
    if _motor_name(old.motor_total_impulse) != _motor_name(new.motor_total_impulse):
        return f"swap motor {_motor_name(old.motor_total_impulse)} → {_motor_name(new.motor_total_impulse)}"
    if old.ballast_mass != new.ballast_mass:
        return f"ballast {old.ballast_mass:.1f} → {new.ballast_mass:.1f} kg"
    if old.rail_inclination != new.rail_inclination:
        return f"rail {old.rail_inclination:.0f}° → {new.rail_inclination:.0f}°"
    if old.parachute_cd_s != new.parachute_cd_s:
        return f"parachute cd·S {old.parachute_cd_s:.1f} → {new.parachute_cd_s:.1f}"
    return "no change"


def _iteration_event(record: dict, opinions: dict, prescreen) -> dict:
    ops = [
        {"role": o.role, "satisfied": o.satisfied,
         "assessment": o.assessment, "suggestion": o.suggestion}
        for o in (opinions[r] for r in ROLES if r in opinions)
    ]
    disagreement = len({o["satisfied"] for o in ops}) > 1  # specialists split
    decision = record["decision"]
    dec = {"kind": decision.kind, "rationale": decision.rationale}
    if decision.kind == "propose" and decision.config is not None:
        dec["change"] = _config_change(record["config"], decision.config)
    ps = None
    if prescreen:
        ps = [{"motor": e.motor_name, "apogee_m": e.predicted_apogee_m,
               "in_envelope": e.in_envelope} for e in prescreen]
    return {
        "type": "iteration",
        "iteration": record["iteration"],
        "config": _config_to_dict(record["config"]),
        "metrics": _metrics_to_dict(record["metrics"]),
        "hard_ok": record["report"].hard_ok,
        "violations": list(record["report"].violations),
        "apogee_gap_m": record["report"].apogee_gap_m,
        "opinions": ops,
        "disagreement": disagreement,
        "prescreen": ps,
        "decision": dec,
    }


def stream_events(mission: Mission, brain: Brain, prescreener=None) -> Iterator[dict]:
    """Run the graph and yield one event dict per iteration, then a verdict event."""
    app = build_graph(mission, brain, prescreener)
    initial = {
        "config": mission.start_config, "iteration": 0, "metrics": None, "report": None,
        "opinions": {}, "prescreen": None, "history": [], "verdict": None,
        "verdict_rationale": None,
    }
    opinions: dict = {}
    prescreen = None
    for update in app.stream(initial, config={"recursion_limit": 100},
                             stream_mode="updates"):
        node, delta = next(iter(update.items()))
        if node == "evaluate":
            opinions, prescreen = {}, None
        elif node in ROLES:
            opinions = delta["opinions"]
        elif node == "prescreen":
            prescreen = delta["prescreen"]
        elif node == "orchestrator":
            record = delta["history"][-1]
            yield _iteration_event(record, opinions, prescreen)
            if "verdict" in delta:
                yield {
                    "type": "verdict",
                    "verdict": delta["verdict"],
                    "rationale": delta["verdict_rationale"],
                    "final_config": _config_to_dict(record["config"]),
                    "final_metrics": _metrics_to_dict(record["metrics"]),
                    "iterations": record["iteration"],
                    "max_iterations": mission.max_iterations,
                }


def collect_run(mission: Mission, brain: Brain, prescreener=None) -> dict:
    """Run to completion and return a full serializable record for recording."""
    return {"mission": mission_to_dict(mission),
            "events": list(stream_events(mission, brain, prescreener))}
