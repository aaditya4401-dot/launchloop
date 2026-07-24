"""
Phase 3 — Step 3c: the Phase 2 apogee model as a cheap PRE-SCREEN.

The oracle (a full RocketPy flight + landing-dispersion ensemble) costs ~5 s.
The Phase 2 apogee model can estimate a config's apogee from just the first 2
seconds of flight — a ~0.1 s ascent sim + a linear model. So we use it to RANK
candidate motors cheaply and let the orchestrator try the most promising one,
instead of spending a full oracle sim on every option. Cheap proposer, expensive
oracle.

CRITICAL SAFEGUARD (per the plan): the apogee model was trained on Phase 1's
thrust range (roughly one motor class ± manufacturing scatter). A proposal whose
early-flight features fall OUTSIDE that training envelope is extrapolation — the
model will be confidently wrong. So:
  - the pre-screen is used ONLY to rank/filter, never to accept a design;
  - every prediction is tagged in_envelope True/False;
  - out-of-envelope estimates are flagged as unreliable (ranking hint only);
  - RocketPy always makes the final call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rocketpy import Flight
from sklearn.linear_model import LinearRegression

from src.agents.mission import MOTOR_MENU
from src.agents.oracle import Config
from src.analysis.features import FEATURE_COLS, WINDOW, build_feature_table
from src.simulate.rocket import (
    HEADING,
    NOMINAL_MASS,
    RAIL_LENGTH,
    build_environment,
    build_motor_for_impulse,
    build_rocket,
)


@dataclass
class PreScreenEstimate:
    motor_name: str
    predicted_apogee_m: float
    in_envelope: bool   # False => extrapolation, ranking hint only (unreliable)


class PreScreener:
    """Trains the Phase 2 apogee model and screens candidate configs cheaply."""

    def __init__(self):
        df = build_feature_table()
        X = df.select(FEATURE_COLS).to_numpy()
        y = df["apogee_agl"].to_numpy()
        self.model = LinearRegression().fit(X, y)
        # Training feature envelope: per-feature min/max the model actually saw.
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)

    def _early_features(self, config: Config, wind_u: float, wind_v: float) -> np.ndarray:
        """Run a cheap ascent-only sim and extract the 4 Phase-2 features.

        Uses max_time just past the feature WINDOW so the sim stops early instead
        of flying to landing — this is what makes the pre-screen cheap.
        """
        env = build_environment(wind_u, wind_v)
        motor = build_motor_for_impulse(config.motor_total_impulse)
        rocket = build_rocket(NOMINAL_MASS, motor,
                              ballast_mass=config.ballast_mass,
                              main_cd_s=config.parachute_cd_s)
        flight = Flight(rocket=rocket, environment=env, rail_length=RAIL_LENGTH,
                        inclination=config.rail_inclination, heading=HEADING,
                        max_time=WINDOW + 0.5, terminate_on_apogee=False)
        t = np.asarray(flight.time)
        tw = t[t <= WINDOW]
        vz = np.asarray(flight.vz(tw))
        az = np.asarray(flight.az(tw))
        # order MUST match FEATURE_COLS = [vz_max, az_max, az_mean, alt_end]
        return np.array([
            float(vz.max()),
            float(az.max()),
            float(az.mean()),
            float(flight.altitude(tw[-1])),
        ])

    def estimate(self, config: Config, wind_u: float, wind_v: float) -> PreScreenEstimate:
        feats = self._early_features(config, wind_u, wind_v)
        pred = float(self.model.predict(feats.reshape(1, -1))[0])
        in_env = bool(np.all(feats >= self.lo) and np.all(feats <= self.hi))
        motor_name = next((m.name for m in MOTOR_MENU
                           if m.total_impulse == config.motor_total_impulse),
                          f"{config.motor_total_impulse:.0f}N·s")
        return PreScreenEstimate(motor_name, pred, in_env)

    def rank_motors(self, base: Config, wind_u: float, wind_v: float,
                    target_apogee: float) -> list[PreScreenEstimate]:
        """Estimate every menu motor (keeping base ballast/rail/chute), best-first
        by closeness to target. In-envelope estimates are trusted over out-of-envelope."""
        import dataclasses
        ests = [
            self.estimate(dataclasses.replace(base, motor_total_impulse=m.total_impulse),
                          wind_u, wind_v)
            for m in MOTOR_MENU
        ]
        # sort: in-envelope first, then by |predicted - target|
        return sorted(ests, key=lambda e: (not e.in_envelope,
                                            abs(e.predicted_apogee_m - target_apogee)))


if __name__ == "__main__":
    # Standalone check: how well does the cheap pre-screen predict each menu motor's
    # apogee, and which fall outside the training envelope? (Compare to the oracle.)
    from src.agents.mission import DEFAULT_MISSION
    from src.agents.oracle import evaluate

    ps = PreScreener()
    m = DEFAULT_MISSION
    print(f"Pre-screen vs oracle @ wind ({m.wind_u:.0f},{m.wind_v:.0f}), nominal ballast/rail\n")
    print(f"{'motor':8s} {'ML pred':>9s} {'envelope':>10s} {'oracle':>8s} {'error':>8s}")
    print("-" * 48)
    for spec in MOTOR_MENU:
        cfg = Config(motor_total_impulse=spec.total_impulse)
        est = ps.estimate(cfg, m.wind_u, m.wind_v)
        truth = evaluate(cfg, m.wind_u, m.wind_v, dispersion_sims=0).apogee_m
        tag = "in" if est.in_envelope else "OUT (⚠)"
        print(f"{spec.name:8s} {est.predicted_apogee_m:9.0f} {tag:>10s} "
              f"{truth:8.0f} {est.predicted_apogee_m - truth:+8.0f}")
    print("\nOut-of-envelope rows are extrapolation — ranking hints only, never accepted.")
