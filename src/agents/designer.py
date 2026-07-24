"""
Phase 3 — Step 3a: the closed-loop designer graph (LangGraph).

Wires the pieces into the propose → evaluate → check → correct loop:

    START → evaluate (RocketPy oracle) → orchestrator (brain decides) → router
              ▲                                                           │
              └───────────────── loop back with the new config ──────────┘
                                              │ verdict OR iteration cap → END

The `brain` is swappable (StubBrain now; ClaudeBrain in 3b) — the graph is
identical either way. RocketPy is always the judge: the orchestrator may only
act on metrics the oracle produced.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TypedDict

import typer
from langgraph.graph import END, START, StateGraph

from src.agents.brain import ROLES, Brain, Opinion, StubBrain
from src.agents.constraints import ConstraintReport, check_constraints
from src.agents.mission import DEFAULT_MISSION, MOTOR_MENU, Mission
from src.agents.oracle import Config, Metrics, evaluate

_IMPULSE_TO_NAME = {m.total_impulse: m.name for m in MOTOR_MENU}


class DesignState(TypedDict):
    config: Config
    iteration: int
    metrics: Metrics | None
    report: ConstraintReport | None
    opinions: dict[str, Opinion]
    prescreen: list | None
    history: list[dict]
    verdict: str | None
    verdict_rationale: str | None


def _motor_name(impulse: float) -> str:
    return _IMPULSE_TO_NAME.get(impulse, f"{impulse:.0f}N·s")


def _fmt_dispersion(d: float | None) -> str:
    return "n/a" if d is None else f"{d:.0f} m"


def _fmt_config(c: Config) -> str:
    return (f"{_motor_name(c.motor_total_impulse)} | ballast {c.ballast_mass:.1f}kg | "
            f"rail {c.rail_inclination:.0f}° | chute cd·S {c.parachute_cd_s:.1f}")


def build_graph(mission: Mission, brain: Brain, prescreener=None):
    """Compile the LangGraph app for a mission + brain (+ optional ML pre-screen)."""

    def evaluate_node(state: DesignState) -> dict:
        it = state["iteration"] + 1
        cfg = state["config"]
        metrics = evaluate(cfg, mission.wind_u, mission.wind_v,
                           dispersion_sims=mission.dispersion_sims)
        report = check_constraints(metrics, mission)
        status = "OK" if report.hard_ok else "VIOLATION"
        print(f"\n[iter {it}] {_fmt_config(cfg)}")
        print(f"         → apogee {metrics.apogee_m:.0f} m | stab "
              f"{metrics.stability_margin_cal:.2f} cal | railV "
              f"{metrics.rail_exit_velocity_ms:.1f} m/s | dispersion "
              f"{_fmt_dispersion(metrics.landing_dispersion_m)}  [{status}]")
        if report.violations:
            for v in report.violations:
                print(f"           - {v}")
        return {"iteration": it, "metrics": metrics, "report": report}

    def make_specialist_node(role: str):
        def specialist_node(state: DesignState) -> dict:
            op = brain.specialist_opinion(role, mission, state["config"],
                                          state["metrics"], state["report"])
            mark = "✓" if op.satisfied else "✗"
            print(f"           [{role:11s} {mark}] {op.assessment}  →  wants: {op.suggestion}")
            return {"opinions": {**state.get("opinions", {}), role: op}}
        return specialist_node

    def prescreen_node(state: DesignState) -> dict:
        # Cheap ML ranking of each menu motor (kept at current ballast/rail/chute).
        ests = prescreener.rank_motors(state["config"], mission.wind_u, mission.wind_v,
                                       mission.target_apogee_m)
        print("           [pre-screen] ML apogee estimates (ranking hint; RocketPy verifies):")
        for e in ests:
            tag = "in-envelope" if e.in_envelope else "OUT-of-envelope ⚠"
            print(f"             {e.motor_name}: ~{e.predicted_apogee_m:.0f} m ({tag})")
        return {"prescreen": ests}

    def orchestrator_node(state: DesignState) -> dict:
        decision = brain.orchestrate(mission, state["config"], state["metrics"],
                                     state["report"], state["opinions"], state["history"],
                                     prescreen=state.get("prescreen"))
        record = {
            "iteration": state["iteration"],
            "config": state["config"],
            "metrics": state["metrics"],
            "report": state["report"],
            "decision": decision,
        }
        upd: dict = {"history": state["history"] + [record]}

        if decision.kind in ("go", "no_go"):
            upd["verdict"] = decision.kind
            upd["verdict_rationale"] = decision.rationale
            print(f"         → orchestrator: {decision.kind.upper()} — {decision.rationale}")
        elif state["iteration"] >= mission.max_iterations:
            upd["verdict"] = "no_go"
            upd["verdict_rationale"] = (
                f"Iteration budget ({mission.max_iterations}) exhausted before "
                f"satisfying constraints. Last idea: {decision.rationale}"
            )
            print(f"         → orchestrator: NO-GO (budget exhausted)")
        else:
            upd["config"] = decision.config
            print(f"         → orchestrator: {decision.rationale}")
        return upd

    def router(state: DesignState):
        return END if state.get("verdict") else "evaluate"

    g = StateGraph(DesignState)
    g.add_node("evaluate", evaluate_node)
    for role in ROLES:
        g.add_node(role, make_specialist_node(role))
    g.add_node("orchestrator", orchestrator_node)

    # evaluate -> performance -> safety -> recovery -> [pre-screen] -> orchestrator
    g.add_edge(START, "evaluate")
    g.add_edge("evaluate", ROLES[0])
    for a, b in zip(ROLES, ROLES[1:]):
        g.add_edge(a, b)
    if prescreener is not None:
        g.add_node("prescreen", prescreen_node)
        g.add_edge(ROLES[-1], "prescreen")
        g.add_edge("prescreen", "orchestrator")
    else:
        g.add_edge(ROLES[-1], "orchestrator")
    g.add_conditional_edges("orchestrator", router, {"evaluate": "evaluate", END: END})
    return g.compile()


def run_design(mission: Mission, brain: Brain, prescreener=None) -> DesignState:
    app = build_graph(mission, brain, prescreener)
    initial: DesignState = {
        "config": mission.start_config,
        "iteration": 0,
        "metrics": None,
        "report": None,
        "opinions": {},
        "prescreen": None,
        "history": [],
        "verdict": None,
        "verdict_rationale": None,
    }
    print("=" * 68)
    print(f"MISSION: reach {mission.target_apogee_m:.0f} m (±{mission.target_tolerance_m:.0f}), "
          f"ceiling {mission.ceiling_m:.0f} m, stability "
          f"{mission.stability_min_cal}-{mission.stability_max_cal} cal, "
          f"rail-exit ≥ {mission.rail_exit_min_ms:.0f} m/s")
    print(f"Wind {mission.wind_u:.0f},{mission.wind_v:.0f} m/s | start: "
          f"{_fmt_config(mission.start_config)} | budget {mission.max_iterations} sims")
    print("=" * 68)

    final = app.invoke(initial, config={"recursion_limit": 100})

    print("\n" + "=" * 68)
    verdict = final["verdict"]
    m = final["metrics"]
    print(f"VERDICT: {verdict.upper()}")
    print(f"  {final['verdict_rationale']}")
    print(f"  Final config : {_fmt_config(final['config'])}")
    print(f"  Final metrics: apogee {m.apogee_m:.0f} m | stab "
          f"{m.stability_margin_cal:.2f} cal | railV {m.rail_exit_velocity_ms:.1f} m/s "
          f"| dispersion {_fmt_dispersion(m.landing_dispersion_m)}")
    print(f"  Iterations   : {final['iteration']} / {mission.max_iterations}")
    print("=" * 68)
    return final


def build_mission(
    target: float | None = None,
    ceiling: float | None = None,
    wind: float | None = None,
    max_iterations: int | None = None,
) -> Mission:
    """DEFAULT_MISSION with any given fields overridden. None means "keep default"."""
    overrides = {}
    if target is not None:
        overrides["target_apogee_m"] = target
    if ceiling is not None:
        overrides["ceiling_m"] = ceiling
    if wind is not None:
        overrides["wind_u"] = wind
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    return dataclasses.replace(DEFAULT_MISSION, **overrides) if overrides else DEFAULT_MISSION


app = typer.Typer(add_completion=False, help="Closed-loop flight designer.")


def _make_brain(kind: str) -> Brain:
    if kind == "stub":
        return StubBrain()
    if kind == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                "OPENAI_API_KEY is not set — the OpenAI agents need it.\n"
                "  export OPENAI_API_KEY=sk-...   then re-run,\n"
                "  or run the offline deterministic version:  make design BRAIN=stub"
            )
        from src.agents.openai_brain import OpenAIBrain
        return OpenAIBrain()
    if kind == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set — the Claude agents need it.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   then re-run,\n"
                "  or run the offline deterministic version:  make design BRAIN=stub"
            )
        from src.agents.claude_brain import ClaudeBrain
        return ClaudeBrain()
    raise SystemExit(f"unknown brain '{kind}' (use 'openai', 'claude', or 'stub')")


@app.command()
def main(
    brain: str = typer.Option("openai", help="Reasoning engine: 'openai' or 'claude' (agents), or 'stub' (offline)."),
    prescreen: bool = typer.Option(True, help="Use the Phase 2 apogee model to pre-screen motor swaps."),
    target: float = typer.Option(None, help="Target apogee in meters (default: 3000)."),
    ceiling: float = typer.Option(None, help="Hard waiver ceiling in meters (default: 3200)."),
    wind: float = typer.Option(None, help="Wind speed in m/s (default: 6)."),
    max_iterations: int = typer.Option(None, help="Simulation budget before a forced no-go (default: 8)."),
):
    """Run the closed-loop designer. Override the default mission with --target/--ceiling/--wind."""
    mission = build_mission(target, ceiling, wind, max_iterations)
    print(f"(brain: {brain}, pre-screen: {prescreen})")
    brain_impl = _make_brain(brain)   # fails fast if claude + no API key
    prescreener = None
    if prescreen:
        from src.agents.prescreen import PreScreener
        print("Training the Phase 2 apogee pre-screen model...")
        prescreener = PreScreener()
    run_design(mission, brain_impl, prescreener)


if __name__ == "__main__":
    app()
