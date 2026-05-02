"""Tests for the matplotlib chart module.

Pure-Python (no Spark): runs fast.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline", "reporting"))

import visualize as viz  # noqa: E402


def _png_signature_ok(path):
    with open(path, "rb") as f:
        head = f.read(8)
    return head[:4] == b"\x89PNG"


def test_category_bar(tmp_path):
    cats = [
        {"label": "toxic",         "count": 100},
        {"label": "severe_toxic",  "count": 5},
        {"label": "obscene",       "count": 50},
        {"label": "threat",        "count": 2},
        {"label": "insult",        "count": 70},
        {"label": "identity_hate", "count": 30},
    ]
    out = viz.chart_category_bar(cats, str(tmp_path / "cat.png"))
    assert os.path.getsize(out) > 1000
    assert _png_signature_ok(out)


def test_distribution_pie(tmp_path):
    out = viz.chart_distribution_pie(
        {"total_scanned": 1000, "total_flagged": 230},
        str(tmp_path / "pie.png"),
    )
    assert os.path.getsize(out) > 1000
    assert _png_signature_ok(out)


def test_distribution_pie_zero_handles_empty(tmp_path):
    out = viz.chart_distribution_pie(
        {"total_scanned": 0, "total_flagged": 0},
        str(tmp_path / "pie_empty.png"),
    )
    assert os.path.exists(out)
    assert _png_signature_ok(out)


def test_user_heatmap(tmp_path):
    rows = [
        {"user_id": 1, "toxic_count": 5, "severe_toxic_count": 0, "obscene_count": 1,
         "threat_count": 0, "insult_count": 2, "identity_hate_count": 0},
        {"user_id": 2, "toxic_count": 1, "severe_toxic_count": 0, "obscene_count": 0,
         "threat_count": 0, "insult_count": 3, "identity_hate_count": 1},
        {"user_id": 7, "toxic_count": 8, "severe_toxic_count": 1, "obscene_count": 4,
         "threat_count": 0, "insult_count": 6, "identity_hate_count": 2},
    ]
    out = viz.chart_user_heatmap(rows, str(tmp_path / "heat.png"), top_n=10)
    assert os.path.getsize(out) > 1000
    assert _png_signature_ok(out)


def test_user_heatmap_empty(tmp_path):
    out = viz.chart_user_heatmap([], str(tmp_path / "heat_empty.png"))
    assert os.path.exists(out)
    assert _png_signature_ok(out)


def test_volume_trend(tmp_path):
    buckets = [
        {"bucket_index": i, "comment_count": 100, "flagged_count": i * 5}
        for i in range(20)
    ]
    out = viz.chart_volume_trend(buckets, str(tmp_path / "vol.png"))
    assert os.path.getsize(out) > 1000
    assert _png_signature_ok(out)


def test_volume_trend_empty(tmp_path):
    out = viz.chart_volume_trend([], str(tmp_path / "vol_empty.png"))
    assert os.path.exists(out)
    assert _png_signature_ok(out)


def test_render_all_writes_four_pngs(tmp_path):
    summary = {"total_scanned": 500, "total_flagged": 100}
    cats = [{"label": l, "count": 10 + i}
            for i, l in enumerate(viz.LABELS)]
    rows = [
        {"user_id": 1, "toxic_count": 5, "severe_toxic_count": 0, "obscene_count": 0,
         "threat_count": 0, "insult_count": 0, "identity_hate_count": 0},
    ]
    buckets = [
        {"bucket_index": i, "comment_count": 50, "flagged_count": 10}
        for i in range(5)
    ]

    out_dir = tmp_path / "out"
    paths = viz.render_all(summary, cats, rows, buckets, str(out_dir))

    assert set(paths.keys()) == {"category_bar", "distribution", "user_heatmap", "volume_trend"}
    for name, p in paths.items():
        assert os.path.exists(p), f"missing {name}"
        assert _png_signature_ok(p), f"{name} is not a valid PNG"
