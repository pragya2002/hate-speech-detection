"""Scalability benchmark for the hate-speech detection pipeline.

Samples the harmonized combined_train parquet at configurable fractions
(default 25%, 50%, 100%), runs the inference and report stages against
each sample, and records per-stage wall time and throughput.

Outputs:
  HDFS  : outputs/benchmark/<run_date>/results            (CSV)
  local : LOCAL_BASE/benchmark/<run_date>/results.csv     (driver-local copy)
  local : LOCAL_BASE/benchmark/<run_date>/scaling_*.png   (charts)

Run:
    spark-submit --master yarn --deploy-mode client \\
        ml-pipeline/benchmarking/run_benchmark.py [fractions]

    fractions: comma-separated, default "0.25,0.5,1.0"

While it runs, watch the Spark UI (link is printed at start) for executor
utilization, shuffle volume, and stage-level CPU. The Spark History Server
preserves the same view after the app finishes.
"""
import os
import sys
import time
import csv
import datetime
import subprocess
from contextlib import contextmanager

from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim, monotonically_increasing_id,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "reporting"))

from config import HDFS_BASE, LOCAL_BASE, LABELS  # noqa: E402
import generate_report as gr  # noqa: E402


@contextmanager
def stopwatch():
    state = {"elapsed": None}
    t0 = time.time()
    try:
        yield state
    finally:
        state["elapsed"] = round(time.time() - t0, 2)


def clean_text(df, text_col="comment_text"):
    return (
        df.withColumn(text_col, lower(col(text_col)))
          .withColumn(text_col, regexp_replace(col(text_col), r"http\S+", ""))
          .withColumn(text_col, regexp_replace(col(text_col), r"[^a-zA-Z\s]", " "))
          .withColumn(text_col, regexp_replace(col(text_col), r"\s+", " "))
          .withColumn(text_col, trim(col(text_col)))
    )


def benchmark_one_scale(spark, fraction, run_date, keep_predictions=False):
    scale_pct = int(round(fraction * 100))
    print(f"\n=== Scale {scale_pct}% ===")

    # ── Stage 1: load + sample ───────────────────────────────────────────────
    with stopwatch() as t:
        full = spark.read.parquet(f"{HDFS_BASE}/data/combined_train")
        sampled = (
            full.sample(fraction=fraction, seed=42)
                .transform(lambda d: clean_text(d))
                .withColumn("id", monotonically_increasing_id().cast("string"))
                .cache()
        )
        n_rows = sampled.count()
    t_load = t["elapsed"]
    print(f"  load_sample : {n_rows:>10,} rows in {t_load:>7}s "
          f"({int(n_rows / max(t_load, 1e-9)):>10,} rows/s)")

    # ── Stage 2: inference (all 6 models) ────────────────────────────────────
    bench_pred_path = f"{HDFS_BASE}/outputs/benchmark/{run_date}/predictions_{scale_pct}"
    with stopwatch() as t:
        output = sampled.select("id", "comment_text")
        for label in LABELS:
            model = PipelineModel.load(f"{HDFS_BASE}/models/{label}_model")
            preds = model.transform(sampled).select(
                "id",
                col("prediction").alias(f"pred_{label}"),
                col("probability").alias(f"prob_{label}"),
            )
            output = output.join(preds, on="id", how="left")
        output.write.mode("overwrite").parquet(bench_pred_path)
    t_inference = t["elapsed"]
    print(f"  inference   : {n_rows:>10,} rows in {t_inference:>7}s "
          f"({int(n_rows / max(t_inference, 1e-9)):>10,} rows/s)")

    # ── Stage 3: report ──────────────────────────────────────────────────────
    with stopwatch() as t:
        pred_df = gr.load_predictions(spark, bench_pred_path)
        pred_df = gr.apply_thresholds(pred_df).cache()
        report_total = pred_df.count()
        flagged_total = pred_df.filter(col("any_flagged")).count()
        gr.build_summary(pred_df, run_date, report_total, flagged_total)
        gr.build_category_breakdown(pred_df, report_total, flagged_total)
        gr.build_flagged_comments(pred_df, run_date).count()
        gr.build_high_risk_users(pred_df, run_date).count()
        gr.build_volume_buckets(pred_df).count()
        pred_df.unpersist()
    t_report = t["elapsed"]
    print(f"  report      : {n_rows:>10,} rows in {t_report:>7}s "
          f"({int(n_rows / max(t_report, 1e-9)):>10,} rows/s)")

    sampled.unpersist()
    if not keep_predictions:
        subprocess.run(
            ["hadoop", "fs", "-rm", "-r", "-skipTrash", bench_pred_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    total = round(t_load + t_inference + t_report, 2)
    rps = lambda secs: int(n_rows / secs) if secs > 0 else 0

    return [
        {"fraction": fraction, "scale_pct": scale_pct, "rows": n_rows,
         "stage": "load_sample", "wall_s": t_load, "rows_per_s": rps(t_load)},
        {"fraction": fraction, "scale_pct": scale_pct, "rows": n_rows,
         "stage": "inference", "wall_s": t_inference, "rows_per_s": rps(t_inference)},
        {"fraction": fraction, "scale_pct": scale_pct, "rows": n_rows,
         "stage": "report", "wall_s": t_report, "rows_per_s": rps(t_report)},
        {"fraction": fraction, "scale_pct": scale_pct, "rows": n_rows,
         "stage": "total", "wall_s": total, "rows_per_s": rps(total)},
    ]


def render_scaling_charts(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    os.makedirs(out_dir, exist_ok=True)
    by_stage = {}
    for r in results:
        by_stage.setdefault(r["stage"], []).append(r)
    for s in by_stage.values():
        s.sort(key=lambda r: r["scale_pct"])

    paths = {}

    # 1. Wall-clock vs. scale (lines per stage)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"load_sample": "#888", "inference": "#d9534f",
              "report": "#5bc0de", "total": "#222"}
    for stage, rows in by_stage.items():
        x = [r["scale_pct"] for r in rows]
        y = [r["wall_s"] for r in rows]
        ax.plot(x, y, marker="o",
                color=colors.get(stage, "#666"),
                linewidth=2 if stage == "total" else 1.5,
                label=stage)
    ax.set_xlabel("Data scale (%)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Pipeline wall time vs. data scale")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out_dir, "scaling_wall_time.png")
    plt.savefig(p, dpi=120)
    plt.close(fig)
    paths["wall_time"] = p

    # 2. Throughput vs. scale
    fig, ax = plt.subplots(figsize=(9, 5))
    for stage, rows in by_stage.items():
        if stage == "total":
            continue
        x = [r["scale_pct"] for r in rows]
        y = [r["rows_per_s"] for r in rows]
        ax.plot(x, y, marker="o",
                color=colors.get(stage, "#666"),
                linewidth=1.5, label=stage)
    ax.set_xlabel("Data scale (%)")
    ax.set_ylabel("Throughput (rows/sec)")
    ax.set_title("Per-stage throughput vs. data scale")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(out_dir, "scaling_throughput.png")
    plt.savefig(p, dpi=120)
    plt.close(fig)
    paths["throughput"] = p

    return paths


def write_local_csv(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["fraction", "scale_pct", "rows", "stage", "wall_s", "rows_per_s"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)


def main():
    fractions = [0.25, 0.5, 1.0]
    if len(sys.argv) > 1:
        fractions = [float(x.strip()) for x in sys.argv[1].split(",") if x.strip()]

    run_date = datetime.date.today().isoformat()

    spark = (SparkSession.builder
             .appName(f"HateSpeech-Benchmark-{run_date}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n=== Scalability Benchmark | {run_date} ===")
    print(f"  Scales        : {fractions}")
    print(f"  Application UI: {spark.sparkContext.uiWebUrl}")
    print(f"  App ID        : {spark.sparkContext.applicationId}")

    all_results = []
    for f in fractions:
        all_results.extend(benchmark_one_scale(spark, f, run_date))

    # Print summary table
    print(f"\n=== Results ===")
    print(f"  {'Scale':>6} {'Rows':>12} {'Stage':<13} {'Wall (s)':>10} {'Rows/sec':>14}")
    print(f"  {'-' * 62}")
    for r in all_results:
        print(f"  {str(r['scale_pct'])+'%':>6} {r['rows']:>12,} {r['stage']:<13} "
              f"{r['wall_s']:>10} {r['rows_per_s']:>14,}")

    # HDFS CSV
    results_df = spark.createDataFrame([Row(**r) for r in all_results])
    hdfs_results = f"{HDFS_BASE}/outputs/benchmark/{run_date}/results"
    results_df.coalesce(1).write.mode("overwrite").option("header", True).csv(hdfs_results)
    print(f"\n  HDFS results : {hdfs_results}")

    # Local CSV + charts
    local_dir = os.path.join(os.path.expanduser(LOCAL_BASE), "benchmark", run_date)
    local_csv = os.path.join(local_dir, "results.csv")
    try:
        write_local_csv(all_results, local_csv)
        print(f"  Local CSV    : {local_csv}")
    except Exception as e:
        print(f"  Local CSV skipped: {e}")

    chart_paths = render_scaling_charts(all_results, local_dir)
    if chart_paths:
        for name, p in chart_paths.items():
            print(f"  Chart {name:<11}: {p}")
    else:
        print(f"  Charts skipped (matplotlib not installed)")

    spark.stop()
    print(f"\n=== Benchmark complete ===")


if __name__ == "__main__":
    main()
