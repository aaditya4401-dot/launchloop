"""
Phase 3 — constraint checking.

Turns raw oracle Metrics into a pass/fail judgment against a Mission's hard
constraints, plus how far the apogee is from the target. This is deterministic
bookkeeping (no LLM): the agents reason ABOUT this report, but the report itself
is ground truth derived from the RocketPy metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.mission import Mission
from src.agents.oracle import Metrics


@dataclass
class ConstraintReport:
    violations: list[str]      # human-readable hard-constraint failures
    apogee_gap_m: float        # apogee - target (negative = short, positive = over)

    @property
    def hard_ok(self) -> bool:
        """True when every hard safety/ceiling constraint is satisfied."""
        return not self.violations

    @property
    def on_target(self) -> bool:
        return abs(self.apogee_gap_m) <= self._tolerance

    @property
    def success(self) -> bool:
        """A flyable design: hard constraints pass AND apogee is on target."""
        return self.hard_ok and self.on_target

    # tolerance is stashed by check_constraints so the properties can use it
    _tolerance: float = 150.0


def check_constraints(metrics: Metrics, mission: Mission) -> ConstraintReport:
    violations: list[str] = []

    if metrics.apogee_m > mission.ceiling_m:
        violations.append(
            f"apogee {metrics.apogee_m:.0f} m exceeds waiver ceiling "
            f"{mission.ceiling_m:.0f} m"
        )
    if not (mission.stability_min_cal <= metrics.stability_margin_cal
            <= mission.stability_max_cal):
        violations.append(
            f"stability {metrics.stability_margin_cal:.2f} cal outside safe band "
            f"{mission.stability_min_cal}-{mission.stability_max_cal} cal"
        )
    if metrics.rail_exit_velocity_ms < mission.rail_exit_min_ms:
        violations.append(
            f"rail-exit velocity {metrics.rail_exit_velocity_ms:.1f} m/s below "
            f"minimum {mission.rail_exit_min_ms:.0f} m/s"
        )

    return ConstraintReport(
        violations=violations,
        apogee_gap_m=metrics.apogee_m - mission.target_apogee_m,
        _tolerance=mission.target_tolerance_m,
    )
