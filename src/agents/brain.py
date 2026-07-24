"""
Phase 3 — the swappable "brain".

The LangGraph graph (src/agents/designer.py) is identical whether the reasoning
comes from a deterministic policy or from Claude. The brain exposes two methods,
one per graph-node role:

    specialist_opinion(role, ...) -> Opinion   # performance / safety / recovery
    orchestrate(..., opinions)    -> Decision   # arbitrate + pick next move

`StubBrain` (this file) is a rules-based stand-in for building/testing the loop
OFFLINE with no API key. In Step 3b, `ClaudeBrain` implements the same two
methods with real Claude calls — and the graph does not change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol

from src.agents.constraints import ConstraintReport
from src.agents.mission import Mission
from src.agents.oracle import Config, Metrics

BALLAST_STEP = 0.5  # kg per corrective ballast change

# The three competing specialist roles.
ROLES = ("performance", "safety", "recovery")


@dataclass
class Opinion:
    """One specialist's read on the current design."""
    role: str
    satisfied: bool       # is this specialist's own objective met?
    assessment: str       # short reasoning
    suggestion: str       # the direction this specialist wants to push


@dataclass
class Decision:
    """The orchestrator's output each turn: a verdict, or the next config to try."""
    kind: str                 # "go" | "no_go" | "propose"
    rationale: str
    config: Config | None = None   # next config to evaluate (kind == "propose")


class Brain(Protocol):
    def specialist_opinion(
        self, role: str, mission: Mission, config: Config,
        metrics: Metrics, report: ConstraintReport,
    ) -> Opinion: ...

    def orchestrate(
        self, mission: Mission, config: Config, metrics: Metrics,
        report: ConstraintReport, opinions: dict[str, Opinion], history: list[dict],
        prescreen: list | None = None,
    ) -> Decision: ...


class StubBrain:
    """Deterministic stand-in for the Claude agents (offline, no API key).

    Specialists give simple rule-based opinions; the orchestrator applies a
    fixed corrective policy (satisfy hard constraints first, then close the gap
    to target), changing one lever per turn so each effect is visible.
    """

    # ---- specialists -----------------------------------------------------
    def specialist_opinion(self, role, mission, config, metrics, report) -> Opinion:
        if role == "performance":
            gap = report.apogee_gap_m
            if gap < -mission.target_tolerance_m:
                return Opinion(role, False,
                               f"Apogee {metrics.apogee_m:.0f} m is {abs(gap):.0f} m short of target.",
                               "increase impulse or shed ballast")
            return Opinion(role, True,
                           f"Apogee {metrics.apogee_m:.0f} m is at/above the target.",
                           "hold or trim slightly")

        if role == "safety":
            if report.hard_ok:
                return Opinion(role, True,
                               f"Ceiling, stability ({metrics.stability_margin_cal:.2f} cal) and "
                               f"rail-exit ({metrics.rail_exit_velocity_ms:.1f} m/s) all within limits.",
                               "hold")
            return Opinion(role, False,
                           "Hard safety/ceiling limits violated: " + "; ".join(report.violations),
                           "reduce apogee / fix stability / raise rail-exit speed")

        # recovery — advisory only (no hard constraint), watches dispersion
        disp = metrics.landing_dispersion_m
        if disp is not None and disp > 400:
            return Opinion(role, False,
                           f"Landing dispersion {disp:.0f} m is large.",
                           "shrink the parachute to cut drift")
        return Opinion(role, True,
                       f"Landing dispersion {'n/a' if disp is None else f'{disp:.0f} m'} acceptable.",
                       "hold")

    # ---- orchestrator ----------------------------------------------------
    def orchestrate(self, mission, config, metrics, report, opinions, history,
                    prescreen=None) -> Decision:
        # StubBrain ignores the ML pre-screen — its policy is deterministic.
        if report.success:
            return Decision("go", (
                f"All hard constraints satisfied and apogee {metrics.apogee_m:.0f} m "
                f"is within {mission.target_tolerance_m:.0f} m of the "
                f"{mission.target_apogee_m:.0f} m target."))

        menu = sorted(mission.motor_menu, key=lambda m: m.total_impulse)
        cur_idx = min(range(len(menu)),
                      key=lambda i: abs(menu[i].total_impulse - config.motor_total_impulse))

        # 1) Apogee over ceiling -> shed altitude: downsize motor, else ballast up.
        if metrics.apogee_m > mission.ceiling_m:
            if cur_idx > 0:
                nxt = menu[cur_idx - 1]
                return self._propose(config, {"motor_total_impulse": nxt.total_impulse},
                    f"Apogee {metrics.apogee_m:.0f} m over ceiling {mission.ceiling_m:.0f} m "
                    f"→ downsize motor {menu[cur_idx].name}→{nxt.name}.")
            return self._propose(config, {"ballast_mass": config.ballast_mass + BALLAST_STEP},
                f"Apogee {metrics.apogee_m:.0f} m over ceiling on smallest motor "
                f"→ add {BALLAST_STEP} kg ballast.")

        # 2) Stability outside the safe band -> adjust ballast toward the band.
        if metrics.stability_margin_cal > mission.stability_max_cal:
            return self._propose(config, {"ballast_mass": max(0.0, config.ballast_mass - BALLAST_STEP)},
                f"Stability {metrics.stability_margin_cal:.2f} cal over "
                f"{mission.stability_max_cal} → remove {BALLAST_STEP} kg ballast.")
        if metrics.stability_margin_cal < mission.stability_min_cal:
            return self._propose(config, {"ballast_mass": config.ballast_mass + BALLAST_STEP},
                f"Stability {metrics.stability_margin_cal:.2f} cal under "
                f"{mission.stability_min_cal} → add {BALLAST_STEP} kg ballast.")

        # 3) Rail-exit too slow -> more impulse (or less ballast).
        if metrics.rail_exit_velocity_ms < mission.rail_exit_min_ms:
            if cur_idx < len(menu) - 1:
                nxt = menu[cur_idx + 1]
                return self._propose(config, {"motor_total_impulse": nxt.total_impulse},
                    f"Rail-exit {metrics.rail_exit_velocity_ms:.1f} m/s below "
                    f"{mission.rail_exit_min_ms:.0f} → upsize motor {menu[cur_idx].name}→{nxt.name}.")
            if config.ballast_mass > 0:
                return self._propose(config, {"ballast_mass": max(0.0, config.ballast_mass - BALLAST_STEP)},
                    f"Rail-exit {metrics.rail_exit_velocity_ms:.1f} m/s low on largest motor "
                    f"→ remove {BALLAST_STEP} kg ballast.")

        # 4) Hard constraints OK but off target -> close the apogee gap.
        if report.apogee_gap_m < -mission.target_tolerance_m:      # too low
            if cur_idx < len(menu) - 1:
                nxt = menu[cur_idx + 1]
                return self._propose(config, {"motor_total_impulse": nxt.total_impulse},
                    f"Apogee {metrics.apogee_m:.0f} m short of target → upsize motor "
                    f"{menu[cur_idx].name}→{nxt.name}.")
            if config.ballast_mass > 0:
                return self._propose(config, {"ballast_mass": max(0.0, config.ballast_mass - BALLAST_STEP)},
                    f"Apogee short on largest motor → remove {BALLAST_STEP} kg ballast.")
        elif report.apogee_gap_m > mission.target_tolerance_m:     # too high, under ceiling
            return self._propose(config, {"ballast_mass": config.ballast_mass + BALLAST_STEP},
                f"Apogee {metrics.apogee_m:.0f} m over target (under ceiling) → add "
                f"{BALLAST_STEP} kg ballast to trim down.")

        return Decision("no_go", (
            "No available corrective action resolves the remaining constraint gap "
            f"(apogee {metrics.apogee_m:.0f} m, violations: {report.violations or 'none'})."))

    @staticmethod
    def _propose(config: Config, changes: dict, rationale: str) -> Decision:
        return Decision("propose", rationale, dataclasses.replace(config, **changes))
