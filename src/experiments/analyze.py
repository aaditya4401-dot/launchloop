"""
Analyze results/study.parquet: paired statistics + plots for the ablation study.

Because every arm sees the SAME missions (RESEARCH_PLAN.md), comparisons are
paired: McNemar's test for the binary success outcome, Wilcoxon signed-rank for
paired apogee error. Reports Wilson 95% CIs on rates. Writes plots to results/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

IN_PATH = Path("results/study.parquet")


def _wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% CI for a proportion (better than normal approx at small n)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def _mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """Paired binary comparison. Returns (a_only, b_only, two-sided p).

    b01 = missions arm A failed but arm B solved; b10 = the reverse. Exact
    McNemar via a binomial test on the discordant pairs.
    """
    b10 = int((a & ~b).sum())   # A solved, B didn't
    b01 = int((~a & b).sum())   # B solved, A didn't
    n = b10 + b01
    p = 1.0 if n == 0 else binomtest(b10, n, 0.5).pvalue
    return b10, b01, p


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"{IN_PATH} not found — run `make study` first.")
    df = pd.read_parquet(IN_PATH)
    arms = list(df["arm"].unique())
    n = len(df) // len(arms)

    print("=" * 70)
    print(f"STUDY ANALYSIS — {len(arms)} arms × {n} missions (paired)")
    print("=" * 70)

    # --- per-arm rates with CIs ---
    print(f"\n{'arm':12s} {'success (95% CI)':>26s} {'hidden viol':>13s} "
          f"{'mean sims':>10s} {'mean |err| m':>13s}")
    print("-" * 78)
    piv = {}
    for arm in arms:
        g = df[df["arm"] == arm].sort_values(
            ["target_apogee_m", "ceiling_m", "wind_ms"]).reset_index(drop=True)
        piv[arm] = g
        k = int(g["success"].sum())
        lo, hi = _wilson(k, n)
        print(f"{arm:12s} {k}/{n}={k/n:5.0%} [{lo:4.0%},{hi:4.0%}]".ljust(38)
              + f"{g['hidden_violation'].mean():11.0%} "
              + f"{g['oracle_sims'].mean():10.2f} "
              + f"{g['apogee_error_m'].mean():12.0f}")

    # --- feasibility from the brute_force ceiling (if present) ---
    feasible = None
    if "brute_force" in piv:
        feasible = piv["brute_force"]["success"].astype(bool).to_numpy()
        print(f"\nFeasibility (brute_force solved): {feasible.sum()}/{n} missions "
              f"({feasible.mean():.0%}) are solvable in the motor×ballast space.")

    # --- paired one_shot vs full_loop, overall and on the feasible subset ---
    if "one_shot" in piv and "full_loop" in piv:
        ga, gb = piv["one_shot"], piv["full_loop"]

        def _compare(mask, label):
            sa = ga["success"].astype(bool).to_numpy()
            sb = gb["success"].astype(bool).to_numpy()
            ea = ga["apogee_error_m"].to_numpy()
            eb = gb["apogee_error_m"].to_numpy()
            if mask is not None:
                sa, sb, ea, eb = sa[mask], sb[mask], ea[mask], eb[mask]
            m = len(sa)
            b10, b01, p = _mcnemar(pd.Series(sa), pd.Series(sb))
            print(f"\n[{label}: {m} missions]  one_shot {sa.mean():.0%} vs "
                  f"full_loop {sb.mean():.0%} success")
            print(f"  McNemar: one_shot-only {b10}, full_loop-only {b01}, p = {p:.3f}")
            try:
                w = wilcoxon(ea, eb)
                print(f"  Wilcoxon (apogee error): p = {w.pvalue:.3f}")
            except ValueError as e:
                print(f"  Wilcoxon (apogee error): n/a ({e})")

        _compare(None, "all")
        if feasible is not None:
            _compare(feasible, "feasible only")

    # --- plots ---
    results_dir = IN_PATH.parent
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    succ = [df[df["arm"] == arm]["success"].mean() for arm in arms]
    hidd = [df[df["arm"] == arm]["hidden_violation"].mean() for arm in arms]
    x = np.arange(len(arms))
    ax[0].bar(x - 0.2, succ, 0.4, label="success", color="#2a9d8f")
    ax[0].bar(x + 0.2, hidd, 0.4, label="hidden violation", color="#e76f51")
    ax[0].set_xticks(x); ax[0].set_xticklabels(arms)
    ax[0].set_ylabel("rate"); ax[0].set_title("Success vs hidden violations")
    ax[0].legend(); ax[0].set_ylim(0, 1)

    ax[1].boxplot([df[df["arm"] == arm]["apogee_error_m"] for arm in arms],
                  tick_labels=arms)
    ax[1].set_ylabel("apogee error (m)"); ax[1].set_title("Distance from target")

    ax[2].boxplot([df[df["arm"] == arm]["oracle_sims"] for arm in arms],
                  tick_labels=arms)
    ax[2].set_ylabel("oracle simulations"); ax[2].set_title("Cost (sims used)")

    fig.tight_layout()
    out = results_dir / "study_summary.png"
    fig.savefig(out, dpi=120)
    print(f"\nWrote plots -> {out}")


if __name__ == "__main__":
    main()
