"""
Phase 3 demo — Streamlit UI for the closed-loop flight designer.

Run it with:  make demo   (i.e. `streamlit run src/demo/app.py`)

Streamlit model in one line: this script re-runs top-to-bottom on every
interaction, and each `st.*` call paints a widget where it executes — so we can
iterate the design loop and render each iteration as it arrives.

Two modes:
  - Replay (default): plays a recorded run from JSON. No API key, never fails.
  - Live: builds a mission from the sliders and runs the real agents (OpenAI).

Both feed the SAME renderer the SAME event dicts (from src/demo/events.py), so
the two modes look identical — one is just pre-recorded.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RECORDED = Path(__file__).parent / "recorded_run.json"
ROLE_ICON = {"performance": "🚀", "safety": "🛡️", "recovery": "🪂"}


# --------------------------------------------------------------------------- #
# Page + sidebar controls                                                     #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Rocket Flight Designer", page_icon="🚀", layout="wide")
st.title("🚀 Closed-loop flight designer")
st.caption(
    "A team of AI agents proposes a rocket configuration, **RocketPy verifies it**, "
    "and they correct and re-simulate until the mission's constraints are met — or "
    "return a justified no-go. Watch where the specialists disagree and where the "
    "simulator overrules them."
)

with st.sidebar:
    st.header("Mission")
    # st.radio returns the selected label; drives which mode we run.
    mode = st.radio(
        "Mode",
        ["Replay (recorded · no key)", "Live (runs the agents)"],
        help="Replay plays a recorded run instantly with no API key. Live runs the "
             "real OpenAI agents (needs OPENAI_API_KEY) — slower, costs a little.",
    )
    live = mode.startswith("Live")

    # Sliders return their current value on every rerun.
    target = st.slider("Target apogee (m)", 1000, 5000, 3000, step=100)
    ceiling = st.slider("Waiver ceiling (m)", 1500, 6000, 3200, step=100)
    wind = st.slider("Wind speed (m/s)", 0, 20, 6, step=1)
    if live and ceiling < target:
        st.warning("Ceiling is below target — the mission may be infeasible.")
    run = st.button("Run design", type="primary", use_container_width=True)
    if not live:
        st.info("Replay mode ignores the sliders — it plays the recorded mission.")


# --------------------------------------------------------------------------- #
# Renderers (shared by both modes)                                            #
# --------------------------------------------------------------------------- #
def render_iteration(ev: dict):
    """Paint one iteration: config → specialist debate → pre-screen → oracle verdict."""
    cfg, mt = ev["config"], ev["metrics"]
    st.markdown(f"### Iteration {ev['iteration']}")
    st.caption(
        f"Trying **{cfg['motor']}** · ballast {cfg['ballast_kg']:.1f} kg · "
        f"rail {cfg['rail_deg']:.0f}° · parachute cd·S {cfg['chute_cd_s']:.1f}"
    )

    # --- the three specialists, side by side (competing objectives) ---
    if ev["disagreement"]:
        st.markdown("**The specialists disagree** ⚔️")
    cols = st.columns(3)
    for col, op in zip(cols, ev["opinions"]):
        mark = "✅" if op["satisfied"] else "❌"
        with col:
            st.markdown(f"**{ROLE_ICON.get(op['role'],'')} {op['role'].title()}** {mark}")
            st.caption(op["assessment"])
            st.caption(f"*wants:* {op['suggestion']}")

    # --- ML pre-screen (ranking hint; out-of-envelope flagged) ---
    if ev["prescreen"]:
        chips = []
        for p in ev["prescreen"]:
            flag = "" if p["in_envelope"] else " ⚠️ out-of-envelope"
            chips.append(f"{p['motor']} ~{p['apogee_m']:.0f} m{flag}")
        st.caption("🔎 ML pre-screen (ranking hint, RocketPy verifies): " + "  ·  ".join(chips))

    # --- the oracle's verdict for this config (the honest judge) ---
    if ev["hard_ok"]:
        st.success(
            f"**RocketPy:** apogee {mt['apogee_m']:.0f} m · stability {mt['stability_cal']:.2f} cal "
            f"· rail-exit {mt['rail_exit_ms']:.1f} m/s — all hard constraints pass"
        )
    else:
        st.error(
            f"**RocketPy overrules:** apogee {mt['apogee_m']:.0f} m — "
            + "; ".join(ev["violations"])
        )

    # --- orchestrator's arbitration ---
    dec = ev["decision"]
    if dec["kind"] == "propose":
        st.markdown(f"🧭 **Orchestrator →** {dec['rationale']}  \n*Next:* `{dec.get('change','')}`")
    else:
        st.markdown(f"🧭 **Orchestrator →** {dec['kind'].upper()}: {dec['rationale']}")
    st.divider()


def render_chart(placeholder, mission: dict, apogees: list[tuple[int, float, bool]], verdict=None):
    """Apogee-vs-iteration convergence, with target & ceiling reference lines."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(mission["ceiling_m"], color="#E45756", ls="--", lw=1.2, label="ceiling (hard)")
    ax.axhline(mission["target_apogee_m"], color="#54A24B", ls="--", lw=1.2, label="target")
    if apogees:
        xs = [a[0] for a in apogees]
        ys = [a[1] for a in apogees]
        ax.plot(xs, ys, "-", color="#4C78A8", lw=1.5, zorder=1)
        ok = [(x, y) for x, y, k in apogees if k]
        bad = [(x, y) for x, y, k in apogees if not k]
        if bad:
            ax.scatter(*zip(*bad), color="#E45756", s=70, zorder=2, label="constraint violated")
        if ok:
            ax.scatter(*zip(*ok), color="#54A24B", s=70, zorder=2, label="all constraints pass")
        ax.set_xticks(xs)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Apogee (m)")
    ax.set_title("Apogee marches toward the target")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)


def render_verdict(ev: dict):
    v = ev["verdict"]
    cfg, mt = ev["final_config"], ev["final_metrics"]
    if v == "go":
        st.success(f"## ✅ GO  ·  {ev['iterations']} / {ev['max_iterations']} simulations")
    else:
        st.error(f"## ⛔ NO-GO  ·  {ev['iterations']} / {ev['max_iterations']} simulations")
    st.write(ev["rationale"])
    c = st.columns(4)
    c[0].metric("Motor", cfg["motor"])
    c[1].metric("Apogee", f"{mt['apogee_m']:.0f} m")
    c[2].metric("Stability", f"{mt['stability_cal']:.2f} cal")
    c[3].metric("Rail-exit", f"{mt['rail_exit_ms']:.1f} m/s")


# --------------------------------------------------------------------------- #
# Event sources                                                               #
# --------------------------------------------------------------------------- #
def replay_source(delay: float = 0.9):
    """Yield recorded events with a small delay so replay feels like a live run."""
    data = json.loads(RECORDED.read_text())
    yield {"type": "mission", **data["mission"]}
    for ev in data["events"]:
        time.sleep(delay)
        yield ev


def live_source(target: int, ceiling: int, wind: int):
    """Build a mission from the sliders and stream the real design loop."""
    import dataclasses
    from src.agents.mission import DEFAULT_MISSION
    from src.agents.openai_brain import OpenAIBrain
    from src.agents.prescreen import PreScreener
    from src.demo.events import mission_to_dict, stream_events

    mission = dataclasses.replace(
        DEFAULT_MISSION, target_apogee_m=float(target), ceiling_m=float(ceiling),
        wind_u=float(wind), wind_v=0.0, dispersion_sims=8,  # fewer sims -> snappier live demo
    )
    yield {"type": "mission", **mission_to_dict(mission)}
    prescreener = PreScreener()          # trains the Phase 2 model (~2 s)
    yield from stream_events(mission, OpenAIBrain(), prescreener)


# --------------------------------------------------------------------------- #
# Drive a run                                                                 #
# --------------------------------------------------------------------------- #
def drive(source):
    """Consume an event source and paint the three panels as events arrive."""
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Design loop")
        loop_box = st.container()
    with right:
        st.subheader("Convergence")
        chart_box = st.empty()

    mission, apogees = None, []
    for ev in source:
        if ev["type"] == "mission":
            mission = ev
            render_chart(chart_box, mission, apogees)
        elif ev["type"] == "iteration":
            with loop_box:
                render_iteration(ev)
            apogees.append((ev["iteration"], ev["metrics"]["apogee_m"], ev["hard_ok"]))
            render_chart(chart_box, mission, apogees)
        elif ev["type"] == "verdict":
            with loop_box:
                render_verdict(ev)


if run:
    if live:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("Live mode needs `OPENAI_API_KEY`. Set it and rerun, or use Replay mode.")
        else:
            with st.spinner("Running the agents against RocketPy…"):
                drive(live_source(target, ceiling, wind))
    else:
        drive(replay_source())
else:
    st.info("Set the mode and press **Run design** in the sidebar. "
            "Replay mode needs no key and plays a recorded run.")
