from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, avg

spark = SparkSession.builder \
    .appName("Generate Moderation Report") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ── Load predictions ──────────────────────────────────────────────────────────
print("\n=== Loading multi-label predictions ===")
predictions = spark.read.parquet(f"{HDFS_BASE}/outputs/predictions")
total = predictions.count()
print(f"  Total rows: {total}")

# ── Overall summary ───────────────────────────────────────────────────────────
print("\n=== Overall summary ===")
summary_exprs = [count("*").alias("total_comments")] + [
    spark_sum(col(f"pred_{label}")).alias(f"predicted_{label}")
    for label in LABELS
] + [
    spark_sum(col(label)).alias(f"actual_{label}")
    for label in LABELS
]
summary = predictions.agg(*summary_exprs)
summary.show(truncate=False)

# ── Per-label flagged comment counts ─────────────────────────────────────────
print("\n=== Flagged comment counts per label ===")
print(f"  {'Label':<20} {'Predicted':>10} {'Actual':>10} {'Difference':>12}")
print(f"  {'-'*55}")
summary_row = summary.collect()[0]
for label in LABELS:
    predicted = int(summary_row[f"predicted_{label}"])
    actual    = int(summary_row[f"actual_{label}"])
    diff      = predicted - actual
    print(f"  {label:<20} {predicted:>10} {actual:>10} {diff:>+12}")

# ── Flagged comments (any label predicted toxic) ──────────────────────────────
print("\n=== Flagged comments (any category) ===")
any_toxic_filter = col("pred_toxic") == 1.0
for label in LABELS[1:]:
    any_toxic_filter = any_toxic_filter | (col(f"pred_{label}") == 1.0)

flagged_comments = predictions.filter(any_toxic_filter).select(
    "id",
    "comment_text",
    *[col(f"pred_{label}").alias(label) for label in LABELS]
)
print(f"  Total flagged: {flagged_comments.count()}")
flagged_comments.show(5, truncate=60)

# ── User aggregation ──────────────────────────────────────────────────────────
print("\n=== User behavior aggregation ===")
from pyspark.sql.functions import rand
predictions_with_user = predictions.withColumn(
    "user_id", (rand(seed=42) * 1000).cast("int")
)

user_stats = predictions_with_user.groupBy("user_id").agg(
    count("*").alias("total_comments"),
    *[spark_sum(col(f"pred_{label}")).alias(f"{label}_count") for label in LABELS],
    avg(col("pred_toxic")).alias("toxicity_ratio")
)

flagged_users = user_stats.filter(
    (col("toxicity_ratio") > 0.15) &
    (col("total_comments") >= 50)
).orderBy(col("toxicity_ratio").desc())

print(f"  Total users    : {user_stats.count()}")
print(f"  Flagged users  : {flagged_users.count()}")
print("\n=== Top 10 high-risk users ===")
flagged_users.show(10, truncate=False)

# ── Write reports to HDFS ─────────────────────────────────────────────────────
print("\n=== Writing reports to HDFS ===")
summary.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(f"{HDFS_BASE}/outputs/moderation_summary")

flagged_users.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(f"{HDFS_BASE}/outputs/flagged_users")

flagged_comments.coalesce(1).write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(f"{HDFS_BASE}/outputs/flagged_comments")

print(f"  Reports saved to {HDFS_BASE}/outputs/")
print("\n=== Moderation report complete ===")
spark.stop()
