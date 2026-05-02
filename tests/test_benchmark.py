"""Tests for the benchmark module's pure-Python helpers.

The Spark-driven benchmark loop (`benchmark_one_scale`, `main`) needs the
cluster's HDFS data + saved models, so it is not unit-tested here. This
suite covers the timing decorator, CSV writer, and chart renderer.
"""
import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline", "reporting"))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline", "benchmarking"))

import run_benchmark as bm  # noqa: E402


def test_stopwatch_records_elapsed():
    import time
    with bm.stopwatch() as t:
        time.sleep(0.05)
    assert t["elapsed"] is not None
    assert 0.04 <= t["elapsed"] <= 1.0


def test_write_local_csv_roundtrip(tmp_path):
    rows = [
        {"fraction": 0.25, "scale_pct": 25, "rows": 1000,
         "stage": "inference", "wall_s": 12.3, "rows_per_s": 81},
        {"fraction": 1.0, "scale_pct": 100, "rows": 4000,
         "stage": "inference", "wall_s": 40.0, "rows_per_s": 100},
    ]
    out = str(tmp_path / "results.csv")
    bm.write_local_csv(rows, out)
    assert os.path.exists(out)
    with open(out) as f:
        reader = csv.DictReader(f)
        loaded = list(reader)
    assert len(loaded) == 2
    assert loaded[0]["scale_pct"] == "25"
    assert loaded[1]["stage"] == "inference"


def test_render_scaling_charts_writes_two_pngs(tmp_path):
    results = []
    for scale in (25, 50, 100):
        n = scale * 1000
        for stage, wall in (("load_sample", scale * 0.05),
                            ("inference", scale * 0.5),
                            ("report", scale * 0.1)):
            results.append({
                "fraction": scale / 100, "scale_pct": scale, "rows": n,
                "stage": stage, "wall_s": round(wall, 2),
                "rows_per_s": int(n / max(wall, 1e-9)),
            })
        total_wall = sum(r["wall_s"] for r in results if r["scale_pct"] == scale)
        results.append({
            "fraction": scale / 100, "scale_pct": scale, "rows": n,
            "stage": "total", "wall_s": round(total_wall, 2),
            "rows_per_s": int(n / max(total_wall, 1e-9)),
        })

    out_dir = tmp_path / "charts"
    paths = bm.render_scaling_charts(results, str(out_dir))
    assert set(paths.keys()) == {"wall_time", "throughput"}
    for name, p in paths.items():
        assert os.path.exists(p), f"missing {name}"
        with open(p, "rb") as f:
            assert f.read(4) == b"\x89PNG"
