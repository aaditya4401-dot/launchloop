# Paper draft — abstract & claims

Working scaffold for a workshop paper / arXiv preprint. Write the claims first:
each is tagged HAVE (evidence exists) or NEED (experiment still to run), so the
remaining work is explicit. Bracketed [TBD] marks numbers that depend on runs
not yet done.

## Working title

*How Much Does the Loop Buy You? An Oracle-Grounded Study of Closed-Loop,
Competing-Objective Agents for Engineering Design*

(Testbed: high-power rocket flight design with RocketPy as the ground-truth
simulator.)

## Abstract (draft)

LLM agents are increasingly proposed for engineering design, but they routinely
assert that a design meets requirements without ever verifying it against a
physical model — and the field has little rigorous measurement of what actually
helps: closing the loop with a simulator, splitting the objective across
competing specialist agents, or screening candidates with a cheap surrogate. We
study these questions on a concrete, reproducible task: designing a high-power
rocket to hit a target apogee under hard safety constraints (a waiver ceiling,
a stability band, a minimum rail-exit velocity), with RocketPy as a ground-truth
oracle that verifies every proposal. A team of competing-objective agents
(performance, safety, recovery) plus an orchestrator proposes one corrective
change at a time under a hard simulation budget. We evaluate on a seeded suite
of randomized missions, bounded below by a strong single-shot surrogate design
and above by exhaustive search over the reachable design space. We find that the
closed loop attains the exhaustive-search feasibility ceiling while using ~5x
fewer simulations, and never ships an invalid design, whereas the single-shot
baseline commits to configurations that the simulator later invalidates on 69%
of missions. [With LLM agents, we further find TBD.] We release the benchmark,
the oracle-grounded evaluation harness, and all baselines.

## Claims (with evidence status)

| # | Claim | Status | Evidence / experiment |
|---|---|---|---|
| C1 | A strong single-shot design commits to configs the simulator later invalidates on ~69% of missions; oracle-in-the-loop verification eliminates these invalid "ships." | **HAVE** | stub, N=100 (one_shot 69% hidden-violation vs full_loop 0%). Note: the 0% is partly by construction; the empirical content is the *measured* 69% one-shot invalid rate. |
| C2 | The closed loop attains the exhaustive-search feasibility ceiling (solves all feasible missions) at ~4.5x fewer simulations (6.3 vs 28). | **HAVE** | stub, N=100 (full_loop 35/35 feasible; brute_force ceiling). Strongest, fully empirical claim. |
| C3 | On feasible missions, the loop's success advantage over a strong one-shot is not statistically significant (honest null / trend). | **HAVE** | N=100: full_loop solved 4 feasible missions one_shot missed, 0 reverse; McNemar p=0.125. Trending, not significant. Do not overclaim. |
| — | **Metric caveat (not a claim):** apogee error must never be reported alone. one_shot has *lower* mean error (385 vs 464 m, Wilcoxon p=0.002) only because it ignores hard constraints (hence its 69% invalid rate); on the feasible subset the direction reverses (p=0.068). Always pair with the violation rate. | **HAVE** | N=100. |
| C4 | LLM agents (Claude/OpenAI) reproduce C1–C2, and by using additional levers (rail angle, parachute) expand the feasible set beyond the motor×ballast ceiling. | **NEED** | Run `full_loop`/`one_shot` with `--brain claude`/`openai`, k=3 repeats, ~20 missions. Also raise brute_force to a richer grid for a fair ceiling. |
| C5 | Competing-objective multi-agent structure outperforms a single agent optimizing everything at once (fewer sims and/or higher success). | **NEED (LLM only)** | The stub ignores specialist opinions, so this is only meaningful with an LLM brain. Add a `single_agent` LLM brain; compare vs full_loop (RQ3). |
| C6 | The ML pre-screen lowers simulations-to-convergence without reducing success. | **NEED (LLM only)** | The stub ignores the pre-screen; only the LLM orchestrator uses it. Run LLM full_loop with `--prescreen` on vs off (RQ4). |
| C7 | C1–C2 hold at larger N with tighter CIs. | **HAVE** | N=100 confirms N=30: CIs [26–45%] for the ceiling arms; C1/C2 unchanged. |

## Experiments still needed (mapped to claims)

1. **LLM arm** (C4) — the highest-value run; needs an API key. Small mission
   subset, k=3 repeats for variance. Decide the ceiling grid (add rail/chute).
2. **single_agent ablation** (C5) — new brain variant, free on stub-style logic.
3. **pre-screen on/off** (C6) — free; flag already wired.
4. **Scale-up** (C7) — N=100+ on the free stub for tight CIs.

## Contribution statement (what the paper offers)

1. A **reproducible, oracle-grounded benchmark** for closed-loop engineering-
   design agents: randomized missions, a real physics oracle, and principled
   floor (one-shot) and ceiling (exhaustive) baselines.
2. An **ablation** isolating the contribution of loop closure, competing-
   objective structure, and surrogate pre-screening.
3. **Honest findings**, including a documented null result, and released code.

## Target venues

- arXiv preprint (achievable now with C1–C3 + benchmark).
- A NeurIPS/ICML/ICLR **workshop** on LLM agents or ML-for-science/engineering
  (achievable with C4 added; C5–C7 strengthen it).

## Threats to validity (carry from RESEARCH_PLAN.md)

Synthetic motor menu + single airframe; RocketPy is both oracle and evaluator
(success is simulator-relative, not flight-tested); modest N; one wind model.
