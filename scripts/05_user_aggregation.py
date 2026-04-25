from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, sum as spark_sum, count, avg

spark = SparkSession.builder \
    .appName("User Toxicity Aggregation") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load predictions output from bridge ───────────────────────────────────────
print("\n=== Loading predictions ===")
df = spark.read.parquet(f"{HDFS_BASE}/outputs/predictions")
print(f"  Rows loaded: {df.count()}")

# Simulate user_id since dataset has no real user IDs
df = df.withColumn("user_id", (rand(seed=42) * 1000).cast("int"))

# ── Aggregate per user ─────────────────────────────────────────────────────────
print("\n=== Aggregating user behavior ===")
user_stats = df.groupBy("user_id").agg(
    count("*").alias("total_comments"),
    spark_sum("predicted_toxic").alias("predicted_toxic_comments"),
    spark_sum("toxic").alias("actual_toxic_comments"),
    spark_sum("severe_toxic").alias("severe_toxic_comments"),
    spark_sum("insult").alias("insult_comments"),
    spark_sum("identity_hate").alias("identity_hate_comments"),
    avg("predicted_toxic").alias("toxicity_ratio")
)

# Flag high-risk users
flagged_users = user_stats.filter(
    (col("toxicity_ratio") > 0.15) &
    (col("total_comments") >= 50)
)

print(f"  Total users: {user_stats.count()}")
print(f"  Flagged high-risk users: {flagged_users.count()}")

print("\n=== Top 10 users by toxicity ratio ===")
user_stats.orderBy(col("toxicity_ratio").desc()).show(10)

print("\n=== Flagged high-risk users ===")
flagged_users.show(10)

# ── Write to HDFS ─────────────────────────────────────────────────────────────
print("\n=== Writing user stats to HDFS ===")
user_stats.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/outputs/user_stats")

flagged_users.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/outputs/flagged_users_parquet")

print("  Saved user_stats and flagged_users to HDFS")
print("\n=== User aggregation complete ===")
spark.stop()
