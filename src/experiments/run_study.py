"""
Ablation study runner (see RESEARCH_PLAN.md).

Runs each arm over the same seeded mission set (a paired comparison) and logs
one row per (mission, arm) to results/study.parquet, then prints a summary.

MVP arms (default, free on the deterministic `stub` brain):
  - one_shot   : single-step baseline (src/experiments/one_shot.py)
  - full_loop  : the real propose->verify->correct designer

Success is ORACLE-defined for every arm: the committed/final config, when
checked by RocketPy, satisfies all hard constraints AND is on target. We also
record `hidden_violation` -- the agent said "go" but the oracle disagrees --
which is the reliability gap the loop is meant to close.

Usage:
    uv run python -m src.experiments.run_study                 # stub, 30 missions
    uv run python -m src.experiments.run_study --n 50
    uv run python -m src.experiments.run_study --brain claude --n 20   # keyed
"""

from __future__ import annotations

import contextlib
import io
import time
from pathlib import Path

import pandas as pd
import typer

from src.agents.constraints import check_constraints
from src.agents.designer import _make_brain, build_graph
from src.agents.mission import Mission
from src.experiments.brute_force import run_brute_force
from src.experiments.missions import random_missions
from src.experiments.one_shot import run_one_shot

OUT_PATH = Path("results/study.parquet")


def _run_full_loop(mission: Mission, brain, prescreener=None) -> dict:
    """Run the real closed-loop designer quietly and return a result row.

    Uses build_graph + invoke directly (not run_design) to avoid its console
    output and its data/last_design.json side effect. The loop already had
    RocketPy evaluate the final config, so we reuse those metrics for the audit
    -- no extra simulation needed.
    """
    app = build_graph(mission, brain, prescreener)
    initial = {
        "config": mission.start_config, "iteration": 0, "metrics": None,
        "report": None, "opinions": {}, "prescreen": None, "history": [],
        "verdict": None, "verdict_rationale": None,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        final = app.invoke(initial, config={"recursion_limit": 100})

    audit_report = check_constraints(final["metrics"], mission)
    agent_verdict = final["verdict"]
    return {
        "arm": "full_loop",
        "agent_verdict": agent_verdict,
        "success": bool(audit_report.success),
        "hidden_violation": agent_verdict == "go" and not audit_report.success,
        "oracle_sims": int(final["iteration"]),
        "final_config": final["config"],
        "audit_metrics": final["metrics"],
        "audit_report": audit_report,
    }


ARMS = {
    "one_shot": lambda mission, brain, pre: run_one_shot(mission, brain),
    "full_loop": _run_full_loop,
    "brute_force": run_brute_force,
}


def _row(mission: Mission, arm: str, res: dict, wall_s: float) -> dict:
    """Flatten a result into a tabular row for the parquet log."""
    m = res["audit_metrics"]
    c = res["final_config"]
    return {
        "arm": arm,
        "target_apogee_m": mission.target_apogee_m,
        "ceiling_m": mission.ceiling_m,
        "wind_ms": mission.wind_u,
        "success": res["success"],
        "hidden_violation": res["hidden_violation"],
        "agent_verdict": res["agent_verdict"],
        "oracle_sims": res["oracle_sims"],
        "final_apogee_m": m.apogee_m,
        "apogee_error_m": abs(m.apogee_m - mission.target_apogee_m),
        "final_stability_cal": m.stability_margin_cal,
        "final_rail_exit_ms": m.rail_exit_velocity_ms,
        "final_dispersion_m": m.landing_dispersion_m,
        "final_motor_impulse": c.motor_total_impulse,
        "final_ballast_kg": c.ballast_mass,
        "wall_s": round(wall_s, 2),
    }


def _summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 68)
    print(f"STUDY SUMMARY  ({df['arm'].nunique()} arms x "
          f"{len(df) // df['arm'].nunique()} missions)")
    print("=" * 68)
    hdr = f"{'arm':12s} {'success':>9s} {'hidden viol':>12s} {'mean sims':>10s} {'mean |err|':>11s}"
    print(hdr); print("-" * len(hdr))
    for arm, g in df.groupby("arm"):
        print(f"{arm:12s} {g['success'].mean():8.0%} "
              f"{g['hidden_violation'].mean():11.0%} "
              f"{g['oracle_sims'].mean():10.2f} "
              f"{g['apogee_error_m'].mean():10.0f}m")
    print("=" * 68)
    print("success      = oracle confirms hard constraints pass AND on target")
    print("hidden viol  = agent declared GO but the oracle disagrees")
    print("mean sims    = oracle evaluations the method used to decide")


app = typer.Typer(add_completion=False, help="Run the closed-loop ablation study.")


@app.command()
def main(
    n: int = typer.Option(30, help="Number of random missions."),
    seed: int = typer.Option(0, help="Mission-set seed (reproducible)."),
    brain: str = typer.Option("stub", help="Reasoning engine: 'stub' (free), 'openai', or 'claude'."),
    prescreen: bool = typer.Option(False, help="Enable the ML pre-screen in the full_loop arm."),
    arms: str = typer.Option("one_shot,full_loop,brute_force", help="Comma-separated arms to run."),
):
    """Run the study matrix and write results/study.parquet."""
    selected = [a.strip() for a in arms.split(",") if a.strip()]
    for a in selected:
        if a not in ARMS:
            raise SystemExit(f"unknown arm '{a}' (choices: {', '.join(ARMS)})")

    missions = random_missions(n, seed)
    brain_impl = _make_brain(brain)
    prescreener = None
    if prescreen:
        from src.agents.prescreen import PreScreener
        prescreener = PreScreener()

    print(f"Study: brain={brain}, arms={selected}, missions={n} (seed {seed})")
    rows: list[dict] = []
    for i, mission in enumerate(missions):
        for arm in selected:
            t0 = time.perf_counter()
            res = ARMS[arm](mission, brain_impl, prescreener)
            rows.append(_row(mission, arm, res, time.perf_counter() - t0))
        done = "".join("✓" if r["success"] else "·"
                       for r in rows[-len(selected):])
        print(f"  mission {i + 1:>3}/{n}  target {mission.target_apogee_m:.0f}m "
              f"wind {mission.wind_u:.0f}  [{'/'.join(selected)}] {done}")

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_PATH}")
    _summary(df)


if __name__ == "__main__":
    app()
