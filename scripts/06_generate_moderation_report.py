from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, count, avg

spark = SparkSession.builder \
    .appName("Generate Moderation Report") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load predictions and user stats from HDFS ─────────────────────────────────
print("\n=== Loading predictions and user stats ===")
predictions = spark.read.parquet(f"{HDFS_BASE}/outputs/predictions")
user_stats  = spark.read.parquet(f"{HDFS_BASE}/outputs/user_stats")
flagged_users = spark.read.parquet(f"{HDFS_BASE}/outputs/flagged_users_parquet")

# ── Overall summary ───────────────────────────────────────────────────────────
print("\n=== Overall dataset summary ===")
summary = predictions.agg(
    count("*").alias("total_comments"),
    spark_sum("predicted_toxic").alias("predicted_toxic"),
    spark_sum("toxic").alias("actual_toxic"),
    spark_sum("severe_toxic").alias("severe_toxic"),
    spark_sum("obscene").alias("obscene"),
    spark_sum("threat").alias("threat"),
    spark_sum("insult").alias("insult"),
    spark_sum("identity_hate").alias("identity_hate")
)
summary.show(truncate=False)

# ── Flagged comments ──────────────────────────────────────────────────────────
print("\n=== Flagged comments (predicted toxic) ===")
flagged_comments = predictions.filter(
    col("predicted_toxic") == 1.0
).select(
    "id",
    "comment_text",
    "predicted_toxic",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
)
print(f"  Total flagged comments: {flagged_comments.count()}")
flagged_comments.show(5, truncate=80)

# ── Top flagged users ─────────────────────────────────────────────────────────
print("\n=== Top 10 high-risk users ===")
flagged_users.orderBy(col("toxicity_ratio").desc()).show(10)

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

print("  Reports saved:")
print(f"  {HDFS_BASE}/outputs/moderation_summary")
print(f"  {HDFS_BASE}/outputs/flagged_users")
print(f"  {HDFS_BASE}/outputs/flagged_comments")

print("\n=== Moderation report complete ===")
spark.stop()
