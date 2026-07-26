# Research plan — turning launchloop into an evaluated study

This turns the working system into a paper-grade result: a research question,
baselines the codebase can already produce, and a concrete experiment protocol.
Target venue: a peer-reviewed **workshop** on LLM agents / ML-for-engineering
(NeurIPS/ICML/ICLR workshops), or an arXiv preprint as a portfolio artifact.

## Research question

> **Does closing the design loop with a simulator oracle — and splitting the
> objective across competing specialist agents — produce more reliable rocket
> designs than weaker baselines, and at what simulation cost?**

Sub-questions (each maps to an ablation the code can run):

- **RQ1 (does the loop matter?)** Full propose→verify→correct loop vs. a
  one-shot design (agent proposes once, no oracle feedback).
- **RQ2 (does grounding matter?)** Loop that acts on RocketPy-verified metrics
  vs. loop that trusts the agents' *self-reported* metrics.
- **RQ3 (does competition matter?)** Three competing specialists + orchestrator
  vs. a single agent optimizing everything at once.
- **RQ4 (does the ML pre-screen help?)** Pre-screen on vs. off — effect on
  simulations-to-convergence and final quality.
- **RQ5 (do LLM agents beat a hand-coded policy?)** `stub` vs. `openai`/`claude`
  brains on the same missions.

## Hypotheses

- H1: the full loop has a substantially higher **success rate** than one-shot.
- H2: oracle-grounded loop has fewer **hidden constraint violations** (designs
  the agent *claims* pass but actually don't) than the self-reported loop.
- H4: the pre-screen lowers **oracle sims to convergence** without lowering
  success rate.

## Experimental arms (all built on existing code)

Everything runs through `run_design(mission, brain, prescreener)` in
`src/agents/designer.py`, which already returns verdict, final config/metrics,
iteration count, and full history. Arms:

| Arm | How | Tests |
|---|---|---|
| `full_loop` | `run_design(..., prescreener)` as-is | the system |
| `no_prescreen` | `run_design(..., prescreener=None)` (existing `--prescreen` flag) | RQ4 |
| `one_shot` | new: brain proposes a config from the mission only; verify once with `evaluate()`; no correction | RQ1 |
| `self_reported` | new: loop where the brain is fed its own asserted metrics instead of the oracle's, oracle used only for the FINAL audit | RQ2 |
| `single_agent` | new brain variant: one agent, no specialist split | RQ3 |
| brain = `stub` / `openai` / `claude` | existing `_make_brain()` | RQ5 |

New code needed is small: a one-shot runner, a self-reported wrapper, and a
single-agent brain. The core loop, oracle, and missions are reused unchanged.

## Test set — randomized missions

Generate **N seeded missions** by varying the fields `build_mission()` already
supports, plus start config:

- `target_apogee_m` ∈ [2500, 3500]
- `ceiling_m` = target + U[100, 600]  (headroom the design must respect)
- `wind` (`wind_u`) ∈ [0, 12] m/s
- fixed: motor menu, stability band, rail-exit floor, `max_iterations = 8`

Fix a random seed so the mission set is reproducible and identical across arms
(paired comparison — every arm sees the same missions).

Start N = 30 for a first result; scale to 50–100 if signal is promising.

## Metrics (logged per run to `results/study.parquet`)

Primary:
- **success** — `verdict == "go"` AND the final config passes the oracle audit.
- **hidden violations** — for `one_shot`/`self_reported`: does the accepted
  design actually violate a hard constraint when RocketPy checks it? (H2)
- **oracle sims used** — `final["iteration"]` (cost).

Secondary:
- final apogee error `|apogee − target|`
- final stability margin, rail-exit velocity (safety headroom)
- landing dispersion
- wall-clock and (for LLM arms) approximate API cost

## Protocol

1. **Stub-first (free, deterministic, reproducible).** Run the whole harness on
   `brain=stub` before spending any API budget. The stub is a real corrective
   policy, so this isolates the *architecture's* value (loop + oracle +
   competition) from LLM variance. A clean headline result — *"even with a
   deterministic policy, closing the loop raises success from X% to Y%"* — costs
   nothing and is fully reproducible.
2. **LLM confirmation.** Then run a smaller set (e.g., 20 missions) on
   `claude`/`openai` to show it holds with real agents. LLM runs are stochastic:
   repeat each mission **k = 3** times, report mean ± std. Stub is deterministic
   (k = 1).
3. **Budget control.** Cost ≈ missions × arms × repeats × sims. Keep the LLM
   matrix small; lean on the stub for the large-N ablations.

## Statistical analysis

- Success rate per arm with a 95% CI; compare arms with a paired test
  (McNemar's, since arms see the same missions).
- Sims-to-convergence: mean ± std; Wilcoxon signed-rank for paired arms.
- With N = 30 this is **preliminary** — say so plainly; it's enough for a
  workshop result, not a definitive benchmark.

## Deliverables

- `src/experiments/missions.py` — seeded random mission generator.
- `src/experiments/one_shot.py` — one-shot designer + oracle audit.
- `src/experiments/single_agent_brain.py` — collapsed single-agent brain.
- `src/experiments/run_study.py` — run the matrix, log rows to
  `results/study.parquet`.
- `notebooks/study_analysis.ipynb` — aggregate, tables, plots (success by arm,
  iterations histogram).
- `Makefile`: `make study` (stub, free) and `make study-llm` (small, keyed).

## Threats to validity (state these honestly in the paper)

- Synthetic motor menu (one scaled base curve), single airframe — narrow design
  space; results may not transfer to a broader catalog.
- RocketPy is the oracle *and* the evaluator, so "success" is
  simulator-relative, not flight-tested.
- Small N; LLM nondeterminism; one wind model.

## Phasing

- **MVP (free):** stub brain, arms {one_shot, full_loop}, N = 30, metrics
  {success, hidden violations, sims}. This alone is a genuine result.
- **+1:** add `no_prescreen` (RQ4) and `single_agent` (RQ3), still stub.
- **+2:** LLM confirmation on a 20-mission subset (RQ5), k = 3.

## Preliminary results (stub brain, N = 30, seed 0)

Arms: `one_shot`, `full_loop`, `brute_force` (exhaustive motor×ballast search =
feasibility ceiling). Reproduce with `make study` then
`uv run python -m src.experiments.analyze`.

| arm | success | feasible-subset | hidden viol | mean sims | mean err |
|---|---|---|---|---|---|
| one_shot | 33% | 83% | 67% | 1.0 | 339 m |
| full_loop | 40% | 100% | 0% | 6.1 | 321 m |
| brute_force | 40% | 100% | 0% | 28.0 | 71 m |

Only 12/30 (40%) missions are feasible in the motor×ballast space. Findings:

1. **Ceiling-matching efficiency (robust).** The closed loop solves every
   feasible mission (12/12), matching exhaustive search, using ~6 simulations
   vs brute-force's 28 (≈4.6× fewer). Same optimum, a fifth of the cost.
2. **Reliability (robust).** The loop never ships an invalid design (0% hidden
   violations); a strong single-shot commits to invalid configs on 67% of
   missions — it cannot detect this without simulating.
3. **Not a claim.** The one_shot-vs-full_loop *success* gap on feasible missions
   (83% vs 100%) is NOT significant at this N (McNemar p = 0.50, 2 discordant).
   Do not claim the loop solves more than one-shot; claim (1) and (2).

Limitations: N = 30; deterministic stub policy (not LLM); feasibility defined
over motor×ballast only (the stub's reachable space — LLM brains can also move
rail/chute, expanding it). These motivate the +1/+2 phases above.
