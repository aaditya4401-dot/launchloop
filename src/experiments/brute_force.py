"""
Exhaustive grid search over the reachable design space — the feasibility oracle
and best-case ceiling for the study.

The corrective policy (and the stub loop) reaches a design by choosing a motor
from the menu and a nose-ballast mass; rail angle and parachute stay at their
defaults. This brute-forces that same motor × ballast grid with the RocketPy
oracle and keeps the configuration that lands closest to target while satisfying
the hard constraints.

Its purpose is twofold:
  1. Feasibility filter — a mission is "feasible" iff some grid config satisfies
     all hard constraints AND is on target. The one_shot vs full_loop comparison
     is then reported on the feasible subset, so it isn't diluted by missions no
     method in this design space could ever solve.
  2. Ceiling — brute_force is the best achievable result in the reachable space,
     an upper bound the agent arms are measured against.

Hard constraints (ceiling, stability, rail-exit) and on-target depend only on the
single deterministic flight, not on landing dispersion, so this runs with
dispersion_sims=0 — making each grid point one flight and the whole search
cheaper per mission than the full loop.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.agents.constraints import check_constraints
from src.agents.mission import Mission
from src.agents.oracle import evaluate

# Ballast points on the stub's 0.5 kg step, up to a generous 3 kg.
BALLAST_GRID = tuple(round(float(b), 2) for b in np.arange(0.0, 3.01, 0.5))


def run_brute_force(mission: Mission, brain=None, prescreener=None) -> dict:
    """Exhaustively search motor × ballast; return the best result row.

    `brain`/`prescreener` are accepted (and ignored) so this matches the arm
    callable signature used by run_study.
    """
    start = mission.start_config

    best_cfg = best_metrics = best_report = None
    best_key = (2, float("inf"))  # (unsatisfied?, apogee error) — lower is better
    solved = False
    sims = 0

    for motor in mission.motor_menu:
        for ballast in BALLAST_GRID:
            cfg = dataclasses.replace(
                start, motor_total_impulse=motor.total_impulse, ballast_mass=ballast)
            metrics = evaluate(cfg, mission.wind_u, mission.wind_v, dispersion_sims=0)
            report = check_constraints(metrics, mission)
            sims += 1
            solved = solved or report.success
            # Prefer a satisfying config; among those, the smallest apogee error.
            key = (0 if report.success else 1, abs(report.apogee_gap_m))
            if key < best_key:
                best_key, best_cfg, best_metrics, best_report = key, cfg, metrics, report

    return {
        "arm": "brute_force",
        "agent_verdict": "go" if solved else "no_go",
        "success": bool(solved),
        "hidden_violation": False,          # only ever reports oracle-confirmed successes
        "oracle_sims": sims,
        "final_config": best_cfg,
        "audit_metrics": best_metrics,
        "audit_report": best_report,
    }
