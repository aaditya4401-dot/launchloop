# Rocket Flight Planner — task commands
# Run `make help` to see what's available.

.PHONY: help install test simulate load transform data dispersion models design demo record all clean

BRAIN ?= openai

help:
	@echo "make install     Install/sync dependencies into .venv (via uv)"
	@echo "make test        Run the pytest suite (~7s, fully offline)"
	@echo "make simulate    Generate 1000 flights -> data/raw/*.parquet + labels (slow, ~6 min)"
	@echo "make data        Rebuild the DuckDB warehouse from the Parquet files (load + dbt)"
	@echo "make dispersion  RocketPy native Monte Carlo dispersion study (standalone)"
	@echo "make models      Phase 2: train apogee prediction + anomaly detection, save charts"
	@echo "make design      Phase 3: closed-loop flight designer (BRAIN=openai default; also claude or stub-offline)"
	@echo "                 override the mission: TARGET=/CEILING=/WIND=/MAX_ITERATIONS="
	@echo "make demo        Streamlit UI for the design loop (replay mode needs no key)"
	@echo "make record      Re-record the demo's replay run to JSON (offline stub)"
	@echo "make all         Full rebuild from scratch: simulate + data"
	@echo "make clean       Remove generated data (Parquet files + warehouse)"

install:
	uv sync

test:
	uv run python -m pytest -v

# --- Phase 1: data engineering ---

# Slow upstream step: run the Monte Carlo flight loop.
simulate:
	uv run python -m src.simulate.run --n 1000 --seed 42

# Load raw Parquet into DuckDB.
load:
	uv run python -m src.pipeline.load

# Transform raw tables into the one-row-per-flight summary (+ run dbt tests).
transform:
	uv run dbt build --project-dir src/dbt --profiles-dir src/dbt

# `make data` rebuilds the warehouse from the existing Parquet files.
data: load transform

# RocketPy's native Monte Carlo dispersion study. Standalone: a demonstration of
# the dispersion feature, deliberately NOT part of the make data pipeline.
dispersion:
	uv run python -m src.simulate.dispersion --n 300

# --- Phase 2: data science ---
# Train both models (apogee regression + anomaly detection) and save charts.
models:
	uv run python -m src.analysis.apogee_model
	uv run python -m src.analysis.anomaly_model

# --- Phase 3: multi-agent AI ---
# Closed-loop flight designer. BRAIN=openai (default) needs OPENAI_API_KEY;
# BRAIN=claude needs ANTHROPIC_API_KEY; BRAIN=stub runs deterministic + offline.
# Override the mission with TARGET=/CEILING=/WIND=/MAX_ITERATIONS= (all optional;
# unset ones keep the default mission's value).
design:
	uv run python -m src.agents.designer --brain $(BRAIN) \
		$(if $(TARGET),--target $(TARGET)) \
		$(if $(CEILING),--ceiling $(CEILING)) \
		$(if $(WIND),--wind $(WIND)) \
		$(if $(MAX_ITERATIONS),--max-iterations $(MAX_ITERATIONS))

# Streamlit demo UI. PYTHONPATH=. so `import src.*` resolves under `streamlit run`.
demo:
	PYTHONPATH=. uv run streamlit run src/demo/app.py

# Re-record the bundled replay run (offline stub by default; for a live OpenAI
# recording run: uv run python -m src.demo.record --brain openai).
record:
	uv run python -m src.demo.record --brain stub

# `make all` regenerates everything from scratch, flights included.
all: simulate data

clean:
	rm -f data/raw/*.parquet data/labels.parquet data/warehouse.duckdb
	rm -rf data/dispersion src/dbt/target src/dbt/logs
