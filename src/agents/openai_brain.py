"""
Phase 3 — the OpenAI-backed brain (default agent backend).

Implements the same `Brain` protocol as StubBrain/ClaudeBrain, using the OpenAI
SDK. The graph in designer.py is unchanged — the three competing specialists and
the orchestrator are the same prompts (from agent_prompts.py); only the API call
differs. Strict JSON via response_format json_schema so the graph acts on the
decision without brittle text parsing.

Model default: gpt-4o (supports strict structured outputs). Change OPENAI_MODEL
or pass model=... to use another (e.g. gpt-4o-mini for cheaper runs).
Requires OPENAI_API_KEY at run time (not at import).
"""

from __future__ import annotations

import json

from openai import OpenAI

from src.agents import agent_prompts as ap
from src.agents.brain import Decision, Opinion

OPENAI_MODEL = "gpt-4o"


class OpenAIBrain:
    """OpenAI-backed specialists + orchestrator (same protocol as StubBrain)."""

    def __init__(self, model: str = OPENAI_MODEL):
        self.model = model
        self.client = OpenAI()  # reads OPENAI_API_KEY at call time

    def _json_call(self, system: str, user: str, schema: dict, name: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        return json.loads(resp.choices[0].message.content)

    def specialist_opinion(self, role, mission, config, metrics, report) -> Opinion:
        data = self._json_call(
            ap.specialist_system(role),
            ap.specialist_user(mission, config, metrics, report),
            ap.OPINION_SCHEMA, "opinion",
        )
        return ap.opinion_from_json(role, data)

    def orchestrate(self, mission, config, metrics, report, opinions, history,
                    prescreen=None) -> Decision:
        data = self._json_call(
            ap.orchestrator_system(),
            ap.orchestrator_user(mission, config, metrics, report, opinions, history, prescreen),
            ap.DECISION_SCHEMA, "decision",
        )
        return ap.decision_from_json(data, config)
