
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

All three phases are built and working: `make data` (Phase 1), `make models` (Phase 2), and `make design` (Phase 3), plus `make dispersion` (standalone) and `make demo` (Streamlit UI for the Phase 3 loop). `PLAN.md` remains the design/scope reference; `README.md` is the walkthrough of the finished system.

## What this project is

A rocket-flight planner & analytics platform that demonstrates three skills in one pipeline: data engineering, data science, and multi-agent AI, applied to simulated rocketry data.

1. **Simulate** thousands of flights with RocketPy (randomized wind, motor strength, mass).
2. **Store & organize** the output into clean, queryable tables (Parquet → DuckDB → dbt).
3. **Learn from it** — scikit-learn models predict apogee and detect anomalous flights.
4. **Design** — a LangGraph team of specialist agents with *competing* objectives (performance, safety, recovery) plus an orchestrator proposes a rocket configuration, verifies it in RocketPy (the oracle), and iterates until the mission's target and safety constraints are met — or returns a justified no-go. The agent backend is swappable: **OpenAI by default** (`gpt-4o`), Anthropic (Claude) optional, or a deterministic offline stub.

## Build order — phase by phase, do not skip ahead

Build in the order below. Each phase must produce real, working output before the next begins — don't scaffold all three phases' directories/code up front.

1. **Phase 1 — data engineering.** RocketPy simulates ~1,000 flights (weather via `custom_atmosphere` only — never the Ensemble/GEFS model, which requires network fetches). ~10% of flights get a deliberately weakened motor (low total impulse) and are labeled as the "bad" answer key for Phase 2. Each flight's time-series (altitude, vertical speed, acceleration) is saved as a Parquet file in `data/raw/`. DuckDB + a small set of dbt models (3-4, not a warehouse) then build one summary table: one row per flight with apogee, max speed, and label. Target command: `make data`.
2. **Phase 2 — data science.** scikit-learn models: (a) predict final apogee from only the first few seconds of each flight, (b) detect anomalies and validate against the Phase 1 weak-motor labels. Charts go in `notebooks/`. Target command: `make models`.
3. **Phase 3 — multi-agent AI (closed-loop flight designer).** Not a report-writer. Given a mission (target apogee, a hard altitude ceiling, the day's wind, a short motor menu), a LangGraph graph designs a flyable configuration: specialist agents with **competing objectives** — **performance** (hit the target apogee), **safety** (stability margin, rail-exit velocity, apogee under the ceiling — directly fights performance), and **recovery** (contain landing dispersion) — plus an **orchestrator** that proposes the next corrective change (ballast, motor swap, rail angle, parachute size), arbitrates the tradeoff, and calls go / no-go. Each proposal is verified by a real RocketPy simulation (**RocketPy is the oracle** — never accept an agent-asserted metric the sim didn't confirm), looping propose → evaluate → check → correct under a **hard iteration cap** (e.g. 8 sims) so cost stays bounded. Emits either a RocketPy-verified configuration or a no-go naming the unmet constraint. A cheap ML pre-screen (the Phase 2 apogee model on a 2-second sim) ranks candidate motors before the full oracle sim — rank-only, flags out-of-envelope proposals, never accepted. **Agent backend is swappable** via `make design BRAIN=openai|claude|stub` (default `openai`, needs `OPENAI_API_KEY`; `stub` is deterministic/offline). A Streamlit UI (`make demo`) streams the loop with a no-key cached-replay default. Target command: `make design`.

## Architecture / data flow

```
RocketPy simulation (random wind/motor/mass per flight)
      ▼
Parquet files in data/raw/        ← one file per flight, full time-series
      ▼
DuckDB + dbt                      ← clean, combine → one-row-per-flight summary
      ├──────────────► scikit-learn models   (predict apogee, detect anomalies)
      ▼
LangGraph agent team ──► propose config ──► RocketPy verifies it (the oracle)
      ▲                                            │
      └──────── adjust & re-simulate ◄─────────────┘   (loop, capped iterations)
      ▼
Verified flight design (+ byproduct report), or a justified no-go
```

The final arrow **loops back**: agents propose, the simulator judges, agents correct — constraint satisfaction against ground truth, not one-shot report generation.

Target repo layout (per `PLAN.md`):

```
src/
├── simulate/   ← RocketPy flight generation
├── pipeline/   ← load Parquet into DuckDB
├── dbt/        ← SQL transformations (clean → summary table)
├── analysis/   ← scikit-learn models
└── agents/     ← LangGraph closed-loop flight designer (competing-objective agents + orchestrator)
data/
├── raw/             ← per-flight Parquet files
└── warehouse.duckdb
notebooks/      ← exploration + charts for the README
```

## Key technical constraints

- **Python 3.13** — avoid 3.14; a mapping feature the project may rely on isn't ready there.
- **Stay fully offline**, except for the LLM API calls in Phase 3 (OpenAI by default; Anthropic optional). Weather must come from RocketPy's `custom_atmosphere` (your own wind numbers). NOTE: the PyPI wheel does **not** bundle RocketPy's example motor/drag files — so motor thrust is defined as an inline `[[time, thrust], ...]` array and drag as a constant coefficient. Never point RocketPy at a downloaded or GitHub example file.
- **Verify the RocketPy API against the installed version** before writing simulation code (`pip show rocketpy`). The project is pinned to **1.12.1** (1.13 was never published to PyPI). Its Monte Carlo / Stochastic module is still evolving — don't assume method/constructor signatures from memory or older docs. (Verified quirks in 1.12.1: a bare scalar stochastic arg means *std dev* with the nominal taken from the base object, so use `(nominal, std)` tuples; and `StochasticRocket` does **not** inherit the base rocket's aerosurfaces or parachutes — they must be re-added wrapped in `Stochastic*` types or the flight is unstable/ballistic.)
- **RocketPy has two distinct data paths — use the right one for each need:**
  - Per-flight *time-series* (altitude/velocity/acceleration over time) needed by the Phase 2 models must be pulled from individual `Flight` objects in a manual simulation loop — `MonteCarlo` alone does not provide this. This manual loop is what feeds the dbt summary table (via per-flight Parquet traces + a labels table).
  - The built-in `MonteCarlo` class exports a dispersion *summary* (mean/std across many randomized flights). It is a **standalone** demonstration (`make dispersion`), deliberately **not** wired into the `make data` pipeline.
  - Do both in Phase 1: a manual loop for time-series traces (feeds `make data`), and the `MonteCarlo` class for the standalone dispersion study.
- **Keep dbt intentionally small** (3-4 models) — it's there to demonstrate the skill, not to build out a real warehouse.

## Definition of done

- Three working commands: `make data`, `make models`, `make design`.
- A README showing the pipeline, an apogee prediction with its error, a caught anomaly, and a sample design run where the agents move a flight to meet its target under a ceiling (with each iteration's RocketPy-verified metrics).
