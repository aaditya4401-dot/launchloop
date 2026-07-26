# Related work

launchloop's Phase 3 is a closed-loop designer: competing-objective LLM agents
propose a change, a physics simulator (RocketPy) verifies it as the ground-truth
oracle, and the loop iterates under a hard cap. That pattern — **generator agents
+ a non-LLM verifier in the decision loop, under a fixed interaction budget** —
is an active area in 2025–2026 agentic-AI research. The project was built
independently; these references are the closest published work and situate its
design.

## Simulator/verifier-in-the-loop design agents

- **Physics-in-the-Loop: A Hybrid Agentic Architecture for Validated CAD
  Engineering Design.** Agents plan → generate → evaluate → revise with
  validated engineering tools embedded directly in the decision loop.
  Closest architectural analogue to `make design`, in the CAD domain.
  https://arxiv.org/html/2605.19717

- **Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering
  Tasks with Generative Optimization.** Formalizes a propose–execute–evaluate
  loop with executable verifier feedback under a *fixed interaction budget* —
  the same shape as launchloop's 8-simulation cap.
  https://arxiv.org/html/2604.12290v1

- **On the Role of Feedback in Test-Time Scaling of Agentic AI Workflows.**
  Discusses the FunSearch/AlphaEvolve pattern: an LLM generator proposes
  candidates and a *deterministic* evaluator scores each against a fitness
  function that does not invoke another LLM. This is exactly launchloop's
  agents-propose / RocketPy-verifies split (never trust an agent-asserted
  metric).
  https://arxiv.org/pdf/2504.01931

## LLM multi-agent frameworks for engineering design

- **TurboAgent: An LLM-Driven Autonomous Multi-Agent Framework for
  Turbomachinery Aerodynamic Design.** LangGraph-based supervisor + specialist
  agents with a physical-validation stage, for jet-engine blade design. Same
  spirit (multi-agent + validation), different domain and no competing-objective
  arbitration.
  https://arxiv.org/pdf/2604.06747

## Foundational tools

- **RocketPy — 6-DOF launch-vehicle trajectory simulator.** The oracle in this
  project; validated to ~1% apogee error against real university flights.
  https://github.com/RocketPy-Team/RocketPy

## How launchloop relates / what appears distinct

The individual ingredients (multi-agent LLMs, simulator verification, ML
surrogates, dbt/DuckDB pipelines) are each established. As of this writing, no
published work or portfolio project was found combining them as launchloop does:

- **RocketPy as a hard oracle** for a **competing-objective** agent team
  (performance vs. safety vs. recovery) with an explicit orchestrator that
  *arbitrates the tradeoff* — rather than a single planner decomposing subtasks.
- A **cheap ML pre-screen** (the Phase 2 apogee surrogate on a 2-second sim)
  used only to *rank* candidate motors before spending a full oracle sim —
  a surrogate-guided candidate ranking step inside the agent loop.
- The agent loop is fed by the project's **own simulated-data pipeline**
  (Parquet → DuckDB → dbt) and ML models, making it end-to-end rather than a
  standalone agent demo.
