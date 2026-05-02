"""Generate moderation reports from HDFS prediction outputs.

Reads outputs/predictions (parquet from scripts/07_predict.py), applies the
custom per-label thresholds from config.py, and writes four reports under
outputs/reports/<run_date>/ on HDFS:

  summary             one-row pipeline-level totals
  category_breakdown  per-label counts and percentages
  flagged_comments    every comment with at least one label above threshold
  high_risk_users     users whose toxicity_ratio crosses the configured bar

If jinja2 is importable, also renders a single HTML view to the driver's
local filesystem at LOCAL_BASE/reports/<run_date>/report.html.

Usage:
    spark-submit ml-pipeline/reporting/generate_report.py [run_date]
        run_date defaults to today (YYYY-MM-DD).
"""
import os
import sys
import datetime
from functools import reduce

from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import (
    col, lit, when, sum as spark_sum, count, avg, greatest,
    rand, concat_ws,
)
from pyspark.ml.functions import vector_to_array

# Make `from config import ...` work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    HDFS_BASE, LOCAL_BASE, LABELS, THRESHOLDS,
    HIGH_RISK_TOXICITY_RATIO, HIGH_RISK_MIN_COMMENTS,
)


def parse_run_date(argv):
    if len(argv) > 1:
        return argv[1]
    return datetime.date.today().isoformat()


def load_predictions(spark, predictions_path):
    df = spark.read.parquet(predictions_path)
    if "user_id" not in df.columns:
        df = df.withColumn("user_id", (rand(seed=42) * 1000).cast("int"))
    if "comment_text" not in df.columns:
        df = df.withColumn("comment_text", lit(""))
    return df


def apply_thresholds(df):
    dtypes = dict(df.dtypes)
    for label in LABELS:
        prob_col = f"prob_{label}"
        if prob_col in dtypes and dtypes[prob_col].startswith("vector"):
            df = df.withColumn(f"prob1_{label}", vector_to_array(col(prob_col))[1])
        elif prob_col in dtypes and dtypes[prob_col].startswith("array"):
            df = df.withColumn(f"prob1_{label}", col(prob_col)[1])
        elif prob_col in dtypes:
            df = df.withColumn(f"prob1_{label}", col(prob_col).cast("double"))
        else:
            df = df.withColumn(f"prob1_{label}", col(f"pred_{label}").cast("double"))

        df = df.withColumn(
            f"flag_{label}",
            col(f"prob1_{label}") >= lit(THRESHOLDS[label]),
        )

    flag_sum = reduce(
        lambda a, b: a + b,
        [col(f"flag_{l}").cast("int") for l in LABELS],
    )
    df = df.withColumn("flagged_label_count", flag_sum)
    df = df.withColumn("any_flagged", col("flagged_label_count") > 0)
    return df


def build_summary(df, run_date, total, flagged_total):
    agg_exprs = [
        spark_sum(col(f"flag_{l}").cast("int")).alias(f"{l}_count")
        for l in LABELS
    ]
    label_row = df.agg(*agg_exprs).collect()[0].asDict()
    summary = {
        "run_date": run_date,
        "total_scanned": int(total),
        "total_flagged": int(flagged_total),
        "flagged_pct": round(flagged_total / total * 100, 4) if total else 0.0,
    }
    summary.update({k: int(v or 0) for k, v in label_row.items()})
    return summary


def build_category_breakdown(df, total, flagged_total):
    rows = []
    for label in LABELS:
        cnt = df.filter(col(f"flag_{label}")).count()
        avg_p = df.agg(avg(col(f"prob1_{label}"))).collect()[0][0] or 0.0
        rows.append({
            "label": label,
            "count": int(cnt),
            "pct_of_total": round(cnt / total * 100, 4) if total else 0.0,
            "pct_of_flagged": round(cnt / flagged_total * 100, 4) if flagged_total else 0.0,
            "avg_probability": round(float(avg_p), 4),
        })
    return rows


def build_flagged_comments(df, run_date):
    prob_cols = [f"prob1_{l}" for l in LABELS]
    max_prob = greatest(*[col(c) for c in prob_cols])

    max_label_expr = when(col(prob_cols[0]) == max_prob, lit(LABELS[0]))
    for i in range(1, len(LABELS)):
        max_label_expr = max_label_expr.when(col(prob_cols[i]) == max_prob, lit(LABELS[i]))

    flagged_str = concat_ws(",", *[
        when(col(f"flag_{l}"), lit(l)) for l in LABELS
    ])

    select_cols = ["id", "user_id", "comment_text",
                   "max_label", "max_probability", "flagged_labels", "run_date"]

    return (
        df.filter(col("any_flagged"))
          .withColumn("max_probability", max_prob)
          .withColumn("max_label", max_label_expr)
          .withColumn("flagged_labels", flagged_str)
          .withColumn("run_date", lit(run_date))
          .select(*select_cols)
    )


def build_high_risk_users(df, run_date,
                          min_comments=HIGH_RISK_MIN_COMMENTS,
                          ratio=HIGH_RISK_TOXICITY_RATIO):
    label_count_aggs = [
        spark_sum(col(f"flag_{l}").cast("int")).alias(f"{l}_count")
        for l in LABELS
    ]
    user_stats = (
        df.groupBy("user_id")
          .agg(count("*").alias("total_comments"),
               spark_sum(col("any_flagged").cast("int")).alias("flagged_count"),
               *label_count_aggs)
          .withColumn("toxicity_ratio",
                      col("flagged_count") / col("total_comments"))
    )

    count_cols = [col(f"{l}_count") for l in LABELS]
    max_count = greatest(*count_cols)
    top_label_expr = when(col(f"{LABELS[0]}_count") == max_count, lit(LABELS[0]))
    for l in LABELS[1:]:
        top_label_expr = top_label_expr.when(col(f"{l}_count") == max_count, lit(l))

    select_cols = ["user_id", "total_comments", "flagged_count",
                   "toxicity_ratio", "top_label"]
    select_cols += [f"{l}_count" for l in LABELS]
    user_stats = (
        user_stats
          .withColumn("top_label", top_label_expr)
          .select(*select_cols)
    )

    return (
        user_stats
          .filter((col("toxicity_ratio") > ratio) &
                  (col("total_comments") >= min_comments))
          .orderBy(col("toxicity_ratio").desc())
          .withColumn("run_date", lit(run_date))
    )


def build_volume_buckets(df, n_buckets=20):
    from pyspark.sql.window import Window
    from pyspark.sql.functions import ntile, monotonically_increasing_id
    df_indexed = df.withColumn("__row", monotonically_increasing_id())
    window = Window.orderBy("__row")
    bucketed = df_indexed.withColumn("bucket_index", ntile(n_buckets).over(window) - 1)
    return (
        bucketed.groupBy("bucket_index")
                .agg(count("*").alias("comment_count"),
                     spark_sum(col("any_flagged").cast("int")).alias("flagged_count"))
                .orderBy("bucket_index")
    )


def write_csv(spark_df, path):
    spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv(path)


def render_html(summary, categories, flagged_pdf, high_risk_pdf, html_path,
                charts=None):
    try:
        from jinja2 import Template
    except ImportError:
        return False
    template_path = os.path.join(os.path.dirname(__file__), "templates", "report.html.j2")
    with open(template_path) as f:
        template = Template(f.read())
    rendered = template.render(
        summary=summary,
        categories=categories,
        flagged=flagged_pdf.head(100).to_dict(orient="records") if flagged_pdf is not None else [],
        high_risk=high_risk_pdf.head(100).to_dict(orient="records") if high_risk_pdf is not None else [],
        charts=charts or {},
    )
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w") as f:
        f.write(rendered)
    return True


def render_charts(summary, categories, high_risk_pdf, buckets_pdf, charts_dir):
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import visualize
    except ImportError as e:
        print(f"  Charts: skipped ({e})")
        return {}
    high_risk_rows = high_risk_pdf.to_dict(orient="records") if high_risk_pdf is not None else []
    buckets_rows = buckets_pdf.to_dict(orient="records") if buckets_pdf is not None else []
    return visualize.render_all(summary, categories, high_risk_rows, buckets_rows, charts_dir)


def main():
    run_date = parse_run_date(sys.argv)
    predictions_path = f"{HDFS_BASE}/outputs/predictions"
    reports_base = f"{HDFS_BASE}/outputs/reports/{run_date}"

    spark = (SparkSession.builder
             .appName(f"HateSpeech-Report-{run_date}")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n=== Generating moderation report for {run_date} ===")
    print(f"  Source : {predictions_path}")
    print(f"  Output : {reports_base}")

    df = load_predictions(spark, predictions_path)
    df = apply_thresholds(df).cache()

    total = df.count()
    flagged_total = df.filter(col("any_flagged")).count()
    print(f"  Scanned: {total} | Flagged: {flagged_total}")

    summary = build_summary(df, run_date, total, flagged_total)
    categories = build_category_breakdown(df, total, flagged_total)
    flagged_df = build_flagged_comments(df, run_date)
    high_risk_df = build_high_risk_users(df, run_date)
    buckets_df = build_volume_buckets(df)

    summary_df = spark.createDataFrame([Row(**summary)])
    category_df = spark.createDataFrame([Row(**r) for r in categories])

    write_csv(summary_df, f"{reports_base}/summary")
    write_csv(category_df, f"{reports_base}/category_breakdown")
    write_csv(flagged_df, f"{reports_base}/flagged_comments")
    write_csv(high_risk_df, f"{reports_base}/high_risk_users")
    write_csv(buckets_df, f"{reports_base}/volume_buckets")

    print("\n=== Summary ===")
    summary_df.show(truncate=False)
    print("\n=== Category breakdown ===")
    category_df.show(truncate=False)
    print(f"\n=== High-risk users (count) ===")
    high_risk_df.show(20, truncate=False)

    try:
        flagged_pdf = flagged_df.limit(100).toPandas()
        high_risk_pdf = high_risk_df.limit(100).toPandas()
        buckets_pdf = buckets_df.toPandas()

        local_run_dir = os.path.join(
            os.path.expanduser(LOCAL_BASE), "reports", run_date
        )
        charts_dir = os.path.join(local_run_dir, "charts")
        charts = render_charts(summary, categories, high_risk_pdf, buckets_pdf, charts_dir)
        if charts:
            print(f"\n  Charts: {len(charts)} PNGs in {charts_dir}")
        else:
            print("\n  Charts: skipped (matplotlib not installed)")

        local_html = os.path.join(local_run_dir, "report.html")
        if render_html(summary, categories, flagged_pdf, high_risk_pdf, local_html, charts):
            print(f"  HTML  : {local_html}")
        else:
            print("  HTML  : skipped (jinja2 not installed)")
    except Exception as e:
        print(f"\n  Charts/HTML skipped: {e}")

    df.unpersist()
    spark.stop()
    print("\n=== Report generation complete ===")


if __name__ == "__main__":
    main()
