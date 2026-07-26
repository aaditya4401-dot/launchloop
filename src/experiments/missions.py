"""
Seeded random missions for the ablation study (see RESEARCH_PLAN.md).

Every arm of the study sees the SAME mission set (a paired comparison), so the
generator is deterministic given a seed. We vary only the fields that define the
design problem's difficulty -- target apogee, ceiling headroom, and wind -- and
keep everything else (motor menu, safety band, rail-exit floor, start config,
iteration budget) at the DEFAULT_MISSION values so the arms differ only in
*method*, not in problem definition.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.agents.mission import DEFAULT_MISSION, Mission


def random_missions(n: int = 30, seed: int = 0) -> list[Mission]:
    """Return `n` reproducible random missions.

    - target_apogee_m ~ U[2500, 3500]
    - ceiling_m       = target + U[100, 600]   (headroom the design must respect)
    - wind_u          ~ U[0, 12] m/s           (crosswind the recovery objective fights)
    """
    rng = np.random.default_rng(seed)
    missions: list[Mission] = []
    for _ in range(n):
        target = float(rng.uniform(2500.0, 3500.0))
        ceiling = target + float(rng.uniform(100.0, 600.0))
        wind = float(rng.uniform(0.0, 12.0))
        missions.append(
            dataclasses.replace(
                DEFAULT_MISSION,
                target_apogee_m=round(target, 1),
                ceiling_m=round(ceiling, 1),
                wind_u=round(wind, 2),
                wind_v=0.0,
            )
        )
    return missions


if __name__ == "__main__":
    print(f"{'#':>3} {'target':>8} {'ceiling':>8} {'wind':>6}")
    print("-" * 28)
    for i, m in enumerate(random_missions()):
        print(f"{i:3d} {m.target_apogee_m:8.0f} {m.ceiling_m:8.0f} {m.wind_u:6.1f}")
