"""End-to-end tests for the moderation report generator.

Builds a small synthetic predictions DataFrame with known answers, then
exercises each public function in ml-pipeline/reporting/generate_report.py.

Run:
    pytest tests/test_report.py -v

Requires: pyspark, pytest. Spawns a local[2] SparkSession.
"""
import os
import sys

import pytest
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline"))
sys.path.insert(0, os.path.join(ROOT, "ml-pipeline", "reporting"))

from config import LABELS  # noqa: E402
import generate_report as gr  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    s = (SparkSession.builder
         .master("local[2]")
         .appName("test_report")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.ui.enabled", "false")
         .getOrCreate())
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def _make_predictions(spark, n=100):
    """Synthesize a predictions DF.

    Pattern: comment i is toxic iff i % 5 == 0. user_id = i % 10.
    So users 0 and 5 see only toxic comments; users 1..4, 6..9 see only clean.
    """
    fields = [
        StructField("id", IntegerType(), False),
        StructField("user_id", IntegerType(), False),
        StructField("comment_text", StringType(), True),
    ]
    for label in LABELS:
        fields.append(StructField(f"pred_{label}", DoubleType(), False))
        fields.append(StructField(f"prob_{label}", VectorUDT(), False))
    schema = StructType(fields)

    rows = []
    for i in range(n):
        is_toxic = (i % 5 == 0)
        row = {
            "id": i,
            "user_id": i % 10,
            "comment_text": f"comment_{i}",
        }
        for label in LABELS:
            if label == "toxic" and is_toxic:
                row[f"pred_{label}"] = 1.0
                row[f"prob_{label}"] = Vectors.dense([0.15, 0.85])
            else:
                row[f"pred_{label}"] = 0.0
                row[f"prob_{label}"] = Vectors.dense([0.95, 0.05])
        rows.append(Row(**row))
    return spark.createDataFrame(rows, schema)


def test_apply_thresholds_flags_toxic(spark):
    df = _make_predictions(spark, 50)
    out = gr.apply_thresholds(df)
    # i = 0,5,10,...,45 -> 10 toxic items in 50
    assert out.filter("flag_toxic = true").count() == 10
    assert out.filter("any_flagged = true").count() == 10
    assert out.filter("flag_threat = true").count() == 0


def test_summary_counts_match(spark):
    df = _make_predictions(spark, 100)
    out = gr.apply_thresholds(df).cache()
    summary = gr.build_summary(out, "2099-01-01",
                               total=100, flagged_total=20)
    assert summary["run_date"] == "2099-01-01"
    assert summary["total_scanned"] == 100
    assert summary["total_flagged"] == 20
    assert summary["flagged_pct"] == 20.0
    assert summary["toxic_count"] == 20
    assert summary["threat_count"] == 0
    assert summary["severe_toxic_count"] == 0


def test_category_breakdown_proportions(spark):
    df = _make_predictions(spark, 100)
    out = gr.apply_thresholds(df).cache()
    cats = gr.build_category_breakdown(out, total=100, flagged_total=20)
    by_label = {c["label"]: c for c in cats}
    assert by_label["toxic"]["count"] == 20
    assert by_label["toxic"]["pct_of_total"] == 20.0
    assert by_label["toxic"]["pct_of_flagged"] == 100.0
    assert by_label["threat"]["count"] == 0
    assert by_label["threat"]["pct_of_total"] == 0.0


def test_flagged_comments_pick_max_label(spark):
    df = _make_predictions(spark, 25)
    out = gr.apply_thresholds(df).cache()
    flagged = gr.build_flagged_comments(out, "2099-01-01")
    pdf = flagged.toPandas()
    # i % 5 == 0 -> i in {0,5,10,15,20} -> 5 flagged
    assert len(pdf) == 5
    assert (pdf["max_label"] == "toxic").all()
    assert pdf["flagged_labels"].iloc[0] == "toxic"
    assert (pdf["max_probability"] >= 0.5).all()


def test_high_risk_users_threshold(spark):
    df = _make_predictions(spark, 100)
    out = gr.apply_thresholds(df).cache()
    high = gr.build_high_risk_users(
        out, "2099-01-01", min_comments=5, ratio=0.5,
    )
    pdf = high.toPandas()
    # Users 0 and 5 see 10/10 toxic comments. Others see 0/10.
    assert set(pdf["user_id"].tolist()) == {0, 5}
    assert (pdf["toxicity_ratio"] == 1.0).all()
    assert (pdf["top_label"] == "toxic").all()


def test_high_risk_users_default_min_excludes_small_users(spark):
    df = _make_predictions(spark, 100)
    out = gr.apply_thresholds(df).cache()
    high = gr.build_high_risk_users(out, "2099-01-01")
    # Default min_comments=50: each synthetic user only has 10 -> none qualify.
    assert high.count() == 0


def test_no_flags_when_clean(spark):
    fields = [
        StructField("id", IntegerType(), False),
        StructField("user_id", IntegerType(), False),
        StructField("comment_text", StringType(), True),
    ]
    for label in LABELS:
        fields.append(StructField(f"pred_{label}", DoubleType(), False))
        fields.append(StructField(f"prob_{label}", VectorUDT(), False))
    schema = StructType(fields)
    rows = []
    for i in range(10):
        row = {"id": i, "user_id": i, "comment_text": "all clean"}
        for label in LABELS:
            row[f"pred_{label}"] = 0.0
            row[f"prob_{label}"] = Vectors.dense([0.99, 0.01])
        rows.append(Row(**row))
    df = spark.createDataFrame(rows, schema)
    out = gr.apply_thresholds(df).cache()
    assert out.filter("any_flagged = true").count() == 0
    summary = gr.build_summary(out, "2099-01-01", total=10, flagged_total=0)
    assert summary["flagged_pct"] == 0.0
    assert all(summary[f"{l}_count"] == 0 for l in LABELS)
