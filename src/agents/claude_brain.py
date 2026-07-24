"""
Phase 3 — Step 3b: the Claude-backed brain.

Implements the same `Brain` protocol as StubBrain (specialist_opinion + orchestrate),
but the reasoning comes from Claude. The graph in designer.py is unchanged — swap
StubBrain -> ClaudeBrain and the propose->evaluate->correct loop runs on real agents.

Design (per the plan's "competing objectives"):
  - Each specialist has ONE objective and a system prompt that tells it to advocate
    only for that objective — performance vs safety are deliberately in tension.
  - The orchestrator sees all three opinions plus the RocketPy-verified metrics and
    arbitrates: it picks the single next corrective change, or calls go / no-go.
  - RocketPy stays the oracle: agents only reason about metrics the simulator
    produced, and every change they propose is verified by a real sim next iteration.

All agents return strict JSON via structured outputs (output_config.format), so the
graph can act on the decision without brittle text parsing. Model: claude-opus-4-8.
Requires ANTHROPIC_API_KEY at run time (not at import).
"""

from __future__ import annotations

import dataclasses
import json

import anthropic

from src.agents.brain import Decision, Opinion
from src.agents.constraints import ConstraintReport
from src.agents.mission import MOTOR_BY_NAME, MOTOR_MENU, Mission
from src.agents.oracle import Config, Metrics

MODEL = "claude-opus-4-8"

_ROLE_OBJECTIVES = {
    "performance": (
        "You are the PERFORMANCE specialist on a rocket flight-design team. Your ONLY "
        "objective is to make the apogee as close as possible to the mission target — "
        "ideally hitting it exactly. You favor more motor impulse and less ballast. You "
        "do NOT worry about the safety ceiling or stability; other specialists cover "
        "those. Advocate purely for performance."
    ),
    "safety": (
        "You are the SAFETY specialist on a rocket flight-design team. Your ONLY "
        "objective is keeping the flight within hard limits: apogee at or under the "
        "waiver ceiling, stability margin inside the safe band, and rail-exit velocity "
        "at or above the minimum. You favor smaller motors and more ballast. You are "
        "deliberately in tension with performance — push for safety margin."
    ),
    "recovery": (
        "You are the RECOVERY specialist on a rocket flight-design team. Your ONLY "
        "objective is a contained landing — small landing dispersion so the rocket is "
        "recoverable and lands in the safe zone. A larger parachute drifts farther in "
        "wind (more dispersion) but lands softer; a smaller one drifts less but lands "
        "harder. Advocate for recovery."
    ),
}

_OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied": {"type": "boolean"},
        "assessment": {"type": "string"},
        "suggestion": {"type": "string"},
    },
    "required": ["satisfied", "assessment", "suggestion"],
    "additionalProperties": False,
}

_MOTOR_NAMES = [m.name for m in MOTOR_MENU]

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["go", "no_go", "propose"]},
        "rationale": {"type": "string"},
        # For "propose": null on a field means "leave it unchanged".
        "motor": {"type": ["string", "null"], "enum": [*_MOTOR_NAMES, None]},
        "ballast_mass_kg": {"type": ["number", "null"]},
        "rail_inclination_deg": {"type": ["number", "null"]},
        "parachute_cd_s": {"type": ["number", "null"]},
    },
    "required": ["action", "rationale", "motor", "ballast_mass_kg",
                 "rail_inclination_deg", "parachute_cd_s"],
    "additionalProperties": False,
}


def _mission_block(mission: Mission) -> str:
    menu = ", ".join(f"{m.name} ({m.total_impulse:.0f} N·s)" for m in MOTOR_MENU)
    return (
        f"MISSION\n"
        f"  target apogee : {mission.target_apogee_m:.0f} m (±{mission.target_tolerance_m:.0f} m counts as hit)\n"
        f"  hard ceiling  : {mission.ceiling_m:.0f} m (apogee must stay at or under)\n"
        f"  stability band: {mission.stability_min_cal}-{mission.stability_max_cal} cal (hard)\n"
        f"  rail-exit min : {mission.rail_exit_min_ms:.0f} m/s (hard)\n"
        f"  day's wind    : {mission.wind_u:.0f}, {mission.wind_v:.0f} m/s\n"
        f"  motor menu    : {menu}\n"
        f"  corrective actions: swap motor (menu), add/remove nose ballast (kg), "
        f"adjust rail inclination (deg, 90=vertical), resize main parachute (cd·S)."
    )


def _state_block(config: Config, metrics: Metrics, report: ConstraintReport) -> str:
    disp = "n/a" if metrics.landing_dispersion_m is None else f"{metrics.landing_dispersion_m:.0f} m"
    motor_name = next((m.name for m in MOTOR_MENU
                       if m.total_impulse == config.motor_total_impulse),
                      f"{config.motor_total_impulse:.0f}N·s")
    violations = "; ".join(report.violations) if report.violations else "none"
    return (
        f"CURRENT CONFIG\n"
        f"  motor {motor_name} | ballast {config.ballast_mass:.1f} kg | "
        f"rail {config.rail_inclination:.0f}° | parachute cd·S {config.parachute_cd_s:.1f}\n"
        f"ROCKETPY-VERIFIED METRICS (ground truth)\n"
        f"  apogee {metrics.apogee_m:.0f} m | stability {metrics.stability_margin_cal:.2f} cal | "
        f"rail-exit {metrics.rail_exit_velocity_ms:.1f} m/s | landing dispersion {disp}\n"
        f"CONSTRAINT CHECK\n"
        f"  hard violations: {violations}\n"
        f"  apogee vs target: {report.apogee_gap_m:+.0f} m"
    )


class ClaudeBrain:
    """Claude-backed specialists + orchestrator (same protocol as StubBrain)."""

    def __init__(self, model: str = MODEL):
        self.model = model
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY at call time

    def _json_call(self, system: str, user: str, schema: dict,
                   effort: str, max_tokens: int) -> dict:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    # ---- specialists -----------------------------------------------------
    def specialist_opinion(self, role, mission, config, metrics, report) -> Opinion:
        system = (
            _ROLE_OBJECTIVES[role]
            + " You are given the current design and its RocketPy-verified metrics. "
            "Judge ONLY from your objective's perspective. `satisfied` = is your own "
            "objective met right now? Keep `assessment` and `suggestion` to one sentence each."
        )
        user = f"{_mission_block(mission)}\n\n{_state_block(config, metrics, report)}"
        data = self._json_call(system, user, _OPINION_SCHEMA, effort="low", max_tokens=2048)
        return Opinion(role=role, satisfied=bool(data["satisfied"]),
                       assessment=data["assessment"], suggestion=data["suggestion"])

    # ---- orchestrator ----------------------------------------------------
    def orchestrate(self, mission, config, metrics, report, opinions, history,
                    prescreen=None) -> Decision:
        system = (
            "You are the ORCHESTRATOR of a rocket flight-design team. Three specialists "
            "(performance, safety, recovery) hold competing objectives; you arbitrate. "
            "RocketPy is the oracle: reason only from the verified metrics given, and "
            "remember any change you propose will be re-simulated next iteration.\n\n"
            "You may also be shown a cheap ML PRE-SCREEN that estimates each motor's "
            "apogee. Use it only to RANK which motor to try — never treat it as truth. "
            "Estimates tagged out-of-envelope are extrapolation and may be badly wrong; "
            "the RocketPy sim next iteration is the real check.\n\n"
            "Decide ONE of:\n"
            "  - go: ALL hard constraints pass AND apogee is within tolerance of target.\n"
            "  - propose: change exactly ONE lever to make progress (set that field, "
            "leave the others null). Prioritize fixing hard-constraint violations, then "
            "close the apogee gap. Don't repeat a configuration already tried.\n"
            "  - no_go: only if no available action can satisfy the constraints.\n"
            "Give a one-sentence rationale that references the tradeoff you arbitrated."
        )
        opinion_txt = "\n".join(
            f"  [{o.role}] satisfied={o.satisfied} — {o.assessment} (wants: {o.suggestion})"
            for o in (opinions[r] for r in ("performance", "safety", "recovery") if r in opinions)
        )
        hist_txt = "  (none yet)"
        if history:
            rows = []
            for h in history:
                m, c = h["metrics"], h["config"]
                mn = next((mm.name for mm in MOTOR_MENU
                           if mm.total_impulse == c.motor_total_impulse), "?")
                rows.append(f"  iter {h['iteration']}: {mn}/ballast{c.ballast_mass:.1f}/"
                            f"rail{c.rail_inclination:.0f} -> apogee {m.apogee_m:.0f} m")
            hist_txt = "\n".join(rows)

        prescreen_txt = ""
        if prescreen:
            rows = "\n".join(
                f"  {e.motor_name}: ~{e.predicted_apogee_m:.0f} m "
                f"({'in-envelope' if e.in_envelope else 'OUT-OF-ENVELOPE, unreliable'})"
                for e in prescreen
            )
            prescreen_txt = (
                "\nML PRE-SCREEN (approximate apogee if you swap to each motor; "
                "ranking hint only, RocketPy verifies)\n" + rows + "\n"
            )

        budget_left = mission.max_iterations - len(history) - 1
        user = (
            f"{_mission_block(mission)}\n\n{_state_block(config, metrics, report)}\n\n"
            f"SPECIALIST OPINIONS\n{opinion_txt}\n{prescreen_txt}\n"
            f"HISTORY (configs already simulated)\n{hist_txt}\n\n"
            f"Simulations remaining after this one: {budget_left}."
        )
        data = self._json_call(system, user, _DECISION_SCHEMA, effort="medium", max_tokens=6000)

        action = data["action"]
        if action in ("go", "no_go"):
            return Decision(kind=action, rationale=data["rationale"])

        changes: dict = {}
        if data["motor"] is not None:
            changes["motor_total_impulse"] = MOTOR_BY_NAME[data["motor"]].total_impulse
        if data["ballast_mass_kg"] is not None:
            changes["ballast_mass"] = max(0.0, float(data["ballast_mass_kg"]))
        if data["rail_inclination_deg"] is not None:
            changes["rail_inclination"] = float(data["rail_inclination_deg"])
        if data["parachute_cd_s"] is not None:
            changes["parachute_cd_s"] = float(data["parachute_cd_s"])
        new_config = dataclasses.replace(config, **changes) if changes else config
        return Decision(kind="propose", rationale=data["rationale"], config=new_config)
