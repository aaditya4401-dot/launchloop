"""
Phase 1 — Step 4a: load the raw Parquet into DuckDB.

Creates data/warehouse.duckdb with two raw tables that dbt then transforms:

  raw_traces  — every flight's time-series, unioned into one table. flight_id is
                recovered from each Parquet's filename (it isn't stored inside the
                trace files), using DuckDB's read_parquet(..., filename=true).
  raw_labels  — one row per flight: the randomization knobs + is_weak_motor +
                the outcome columns recorded during simulation.

This is deliberately thin: it just lands the files in the database as-is. All
cleaning/combining happens in dbt (the models/ next door).
"""

from pathlib import Path

import duckdb

WAREHOUSE = Path("data/warehouse.duckdb")
TRACES_GLOB = "data/raw/flight_*.parquet"
LABELS_PATH = "data/labels.parquet"


def load() -> None:
    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE))

    # raw_traces: union all per-flight traces; parse flight_id from the filename.
    con.execute("DROP TABLE IF EXISTS raw_traces")
    con.execute(
        r"""
        CREATE TABLE raw_traces AS
        SELECT
            CAST(regexp_extract(filename, 'flight_(\d+)\.parquet', 1) AS INTEGER) AS flight_id,
            time,
            altitude_agl,
            vertical_velocity,
            vertical_accel
        FROM read_parquet(?, filename = true)
        """,
        [TRACES_GLOB],
    )

    # raw_labels: one row per flight, straight from the labels Parquet.
    con.execute("DROP TABLE IF EXISTS raw_labels")
    con.execute(
        "CREATE TABLE raw_labels AS SELECT * FROM read_parquet(?)",
        [LABELS_PATH],
    )

    n_traces = con.execute("SELECT COUNT(*) FROM raw_traces").fetchone()[0]
    n_flights = con.execute("SELECT COUNT(DISTINCT flight_id) FROM raw_traces").fetchone()[0]
    n_labels = con.execute("SELECT COUNT(*) FROM raw_labels").fetchone()[0]
    con.close()

    print(f"Loaded into {WAREHOUSE}")
    print(f"  raw_traces : {n_traces:,} rows across {n_flights:,} flights")
    print(f"  raw_labels : {n_labels:,} rows (one per flight)")


if __name__ == "__main__":
    load()
