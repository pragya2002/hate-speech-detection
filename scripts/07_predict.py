from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, lower, regexp_replace, trim, monotonically_increasing_id
import time

spark = SparkSession.builder \
    .appName("HateSpeech-MultiLabel-Predict") \
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

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/combined_train")
df = df.filter(col("comment_text").isNotNull())
print(f"  Rows loaded: {df.count()}")

# ── Apply NLP cleaning ────────────────────────────────────────────────────────
print("\n=== Applying NLP preprocessing ===")
cleaned = df \
    .withColumn("comment_text", lower(col("comment_text"))) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"http\S+", "")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"\s+", " ")) \
    .withColumn("comment_text", trim(col("comment_text")))

# ── Run all 6 models ──────────────────────────────────────────────────────────
print("\n=== Running multi-label predictions ===")

cleaned = cleaned.withColumn("id", monotonically_increasing_id().cast("string"))

output = cleaned.select(
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
)

for label in LABELS:
    print(f"  Predicting: {label}")
    t0 = time.time()
    model = PipelineModel.load(f"{HDFS_BASE}/models/{label}_model")
    preds = model.transform(cleaned).select(
        "id",
        col("prediction").alias(f"pred_{label}"),
        col("probability").alias(f"prob_{label}")
    )
    output = output.join(preds, on="id", how="left")
    print(f"  Done in {round(time.time()-t0, 2)}s")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n=== Prediction summary ===")
print(f"  {'Label':<20} {'Flagged':>10} {'% of total':>12}")
print(f"  {'-'*45}")
total = output.count()
for label in LABELS:
    flagged = output.filter(col(f"pred_{label}") == 1.0).count()
    print(f"  {label:<20} {flagged:>10} {flagged/total*100:>11.2f}%")

# ── Write to HDFS ─────────────────────────────────────────────────────────────
print("\n=== Writing multi-label predictions to HDFS ===")
output.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/outputs/predictions")

print(f"  Saved {total} rows to {HDFS_BASE}/outputs/predictions")
print("\n=== Prediction complete ===")
spark.stop()
