"""
Phase 3 — Step 2: the motor menu (and, later, the mission definition).

The agents can't choose a motor if there's only one. This defines a short menu
of solid motors, each just the base M1670 thrust curve scaled to a different
total impulse (per the plan). Names follow real rocketry convention:
letter class + average thrust, e.g. "M1540" = M-class, ~1540 N average thrust.

Impulse classes (N·s):  K 1280-2560 · L 2560-5120 · M 5120-10240.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.oracle import Config
from src.simulate.rocket import BURN_TIME


@dataclass(frozen=True)
class MotorSpec:
    name: str
    total_impulse: float  # N·s

    @property
    def avg_thrust(self) -> float:
        return self.total_impulse / BURN_TIME


# The available motors, small -> large. 9000 N·s is the mission's starting motor
# (deliberately over the ceiling); 6000 N·s is the sweet spot near target.
MOTOR_MENU: list[MotorSpec] = [
    MotorSpec("L900", 3500.0),   # L class — well under target on its own
    MotorSpec("M1540", 6000.0),  # M class — lands near the 3000 m target
    MotorSpec("M1920", 7500.0),  # M class — over the ceiling
    MotorSpec("M2310", 9000.0),  # M class — mission start, well over the ceiling
]

MOTOR_BY_NAME = {m.name: m for m in MOTOR_MENU}


@dataclass(frozen=True)
class Mission:
    """The design problem: hit the target apogee under hard safety limits.

    Defaults are calibrated to our Phase 1 numbers (nominal ~3300 m) so the
    STARTING config (the biggest motor, ~5000 m) violates the ceiling — the
    agents must actually downsize / ballast to succeed, not pass on turn one.
    """
    target_apogee_m: float = 3000.0
    ceiling_m: float = 3200.0            # hard waiver ceiling
    stability_min_cal: float = 1.5       # hard safety band (lower)
    stability_max_cal: float = 2.5       # hard safety band (upper)
    rail_exit_min_ms: float = 20.0       # hard safety floor
    target_tolerance_m: float = 150.0    # how close to target counts as "hit"

    wind_u: float = 6.0                  # the day's wind (m/s)
    wind_v: float = 0.0

    max_iterations: int = 8              # hard cap on RocketPy evaluations
    dispersion_sims: int = 16            # per-evaluate landing-dispersion ensemble

    # Available motors and the starting configuration (biggest motor).
    motor_menu: tuple[MotorSpec, ...] = field(default_factory=lambda: tuple(MOTOR_MENU))
    start_config: Config = field(
        default_factory=lambda: Config(motor_total_impulse=MOTOR_BY_NAME["M2310"].total_impulse)
    )


DEFAULT_MISSION = Mission()


if __name__ == "__main__":
    print("Motor menu (base M1670 curve scaled to each impulse):\n")
    print(f"{'name':8s} {'impulse (N·s)':>14s} {'avg thrust (N)':>15s}")
    print("-" * 40)
    for m in MOTOR_MENU:
        print(f"{m.name:8s} {m.total_impulse:14.0f} {m.avg_thrust:15.0f}")
