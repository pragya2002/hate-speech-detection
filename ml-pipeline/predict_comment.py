from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, lower, regexp_replace, trim
import sys

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

# Get comment from command line
comment = " ".join(sys.argv[1:])
print(f"\n=== Input Comment ===")
print(f"  \"{comment}\"")

# ── Clean comment ─────────────────────────────────────────────────────────────
data = [(1, comment)]
df = spark.createDataFrame(data, ["id", "comment_text"])
cleaned = df \
    .withColumn("comment_text", lower(col("comment_text"))) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"http\S+", "")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"\s+", " ")) \
    .withColumn("comment_text", trim(col("comment_text")))

# ── Load all 6 models and predict ─────────────────────────────────────────────
print("\n=== Prediction Results ===")
print(f"  {'Label':<20} {'Probability':>12} {'Verdict':>10}")
print(f"  {'-'*45}")

flagged = []
for label in LABELS:
    model = PipelineModel.load(f"{HDFS_BASE}/models/{label}_model")
    prediction = model.transform(cleaned)
    row = prediction.select("prediction", "probability").collect()[0]
    prob = round(float(row["probability"][1]) * 100, 2)
    verdict = "🚨 YES" if row["prediction"] == 1.0 else "✅ NO"
    print(f"  {label:<20} {prob:>11}% {verdict:>10}")
    if row["prediction"] == 1.0:
        flagged.append(label)

# ── Overall verdict ───────────────────────────────────────────────────────────
print(f"\n=== Overall Verdict ===")
if flagged:
    print(f"  🚨 TOXIC COMMENT DETECTED")
    print(f"  Categories : {', '.join(flagged)}")
else:
    print(f"  ✅ CLEAN COMMENT")

print("\n=== Done ===")
spark.stop()
