from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, lower, regexp_replace, trim

spark = SparkSession.builder \
    .appName("HateSpeech-Predict") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load raw data from HDFS ───────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")

# ── Apply Pragya's cleaning steps ─────────────────────────────────────────────
print("\n=== Applying NLP preprocessing ===")
cleaned_df = df \
    .withColumn("comment_text", lower(col("comment_text"))) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"http\S+", "")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"\s+", " ")) \
    .withColumn("comment_text", trim(col("comment_text"))) \
    .filter(col("comment_text").isNotNull())

print(f"  Rows after cleaning: {cleaned_df.count()}")

# ── Load your saved final model ───────────────────────────────────────────────
print("\n=== Loading final model ===")
model = PipelineModel.load(f"{HDFS_BASE}/models/final_model")

# ── Run predictions ───────────────────────────────────────────────────────────
print("\n=== Running predictions ===")
predictions = model.transform(cleaned_df)

# Keep only what downstream scripts need
output = predictions.select(
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    col("prediction").alias("predicted_toxic"),
    col("probability")
)

print(f"  Total predictions: {output.count()}")
print(f"  Flagged as toxic : {output.filter(col('predicted_toxic') == 1.0).count()}")

# ── Write predictions to HDFS ─────────────────────────────────────────────────
print("\n=== Writing predictions to HDFS ===")
output.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/outputs/predictions")

print(f"  Saved to {HDFS_BASE}/outputs/predictions")

# ── Verify ────────────────────────────────────────────────────────────────────
verify = spark.read.parquet(f"{HDFS_BASE}/outputs/predictions")
print(f"\n=== Verification: {verify.count()} rows in predictions output ===")
verify.printSchema()

print("\n=== Prediction complete ===")
spark.stop()
