"""
Tests for the Phase 2 feature pipeline (src/analysis/features.py).

Builds two REAL flights (one normal, one weak-motor) with the actual RocketPy
simulator into a temp directory, so the DuckDB query is exercised against real
trace data rather than mocked -- without depending on the full 1,000-flight
dataset the real pipeline runs against.
"""

import polars as pl

from src.analysis.features import FEATURE_COLS, build_feature_table
from src.simulate.rocket import ELEVATION, build_flight
from src.simulate.trace import save_trace


def _make_flight_fixture(tmp_path, flight_id: int, thrust_scale: float) -> float:
    """Fly one real flight and save its trace; return its apogee AGL (m)."""
    flight = build_flight(wind_u=5.0, wind_v=0.0, mass=14.426, thrust_scale=thrust_scale)
    save_trace(flight, tmp_path / f"flight_{flight_id:04d}.parquet")
    return float(flight.apogee) - ELEVATION


def test_feature_table_schema_and_no_nulls(tmp_path):
    apogee_normal = _make_flight_fixture(tmp_path, 1, thrust_scale=1.0)
    apogee_weak = _make_flight_fixture(tmp_path, 2, thrust_scale=0.6)  # sabotaged motor

    labels = pl.DataFrame({
        "flight_id": [1, 2],
        "apogee_agl": [apogee_normal, apogee_weak],
        "is_weak_motor": [False, True],
    })
    labels_path = tmp_path / "labels.parquet"
    labels.write_parquet(labels_path)

    df = build_feature_table(
        traces_glob=str(tmp_path / "flight_*.parquet"),
        labels_path=str(labels_path),
    )

    assert df.height == 2
    assert set(FEATURE_COLS) <= set(df.columns)
    assert df.null_count().sum_horizontal().sum() == 0

    normal_row = df.filter(pl.col("flight_id") == 1)
    weak_row = df.filter(pl.col("flight_id") == 2)
    # The sabotaged motor should show a weaker early-flight signal.
    assert weak_row["vz_max"][0] < normal_row["vz_max"][0]


def test_feature_cols_have_no_collinear_duplicate():
    # vz_end was deliberately dropped (see features.py's docstring): velocity is
    # still rising through the whole window, so it's identical to vz_max and
    # would corrupt the linear-regression coefficients.
    assert "vz_end" not in FEATURE_COLS
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
