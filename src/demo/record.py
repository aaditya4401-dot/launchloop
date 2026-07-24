"""
Phase 3 demo — recorder.

Runs the design loop once and writes the events to JSON, which the Streamlit app
replays with no API key. Defaults to the deterministic `stub` brain so a valid
recording can be produced offline; record a live run to make the default replay
show real LLM reasoning:

    uv run python -m src.demo.record --brain stub      # offline, no key
    uv run python -m src.demo.record --brain openai    # real agents (needs key)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from src.agents.designer import _make_brain
from src.agents.mission import DEFAULT_MISSION
from src.agents.prescreen import PreScreener
from src.demo.events import collect_run

OUT = Path("src/demo/recorded_run.json")
app = typer.Typer(add_completion=False)


@app.command()
def main(brain: str = "stub", out: str = str(OUT)):
    """Record one design run to JSON for the demo's replay mode."""
    brain_impl = _make_brain(brain)   # fails fast if a live brain has no key
    print("Training the Phase 2 apogee pre-screen model...")
    prescreener = PreScreener()
    print(f"Recording a '{brain}' design run...")
    record = collect_run(DEFAULT_MISSION, brain_impl, prescreener)
    Path(out).write_text(json.dumps(record, indent=2))
    verdict = record["events"][-1].get("verdict", "?")
    print(f"Wrote {out} — {len(record['events'])} events, verdict = {verdict.upper()}")


if __name__ == "__main__":
    app()
