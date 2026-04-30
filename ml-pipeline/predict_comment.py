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

# Custom thresholds per label based on class rarity and model behavior
THRESHOLDS = {
    "toxic":         0.50,
    "severe_toxic":  0.80,  # rare class — model outputs near-constant ~55%, raise bar
    "obscene":       0.50,
    "threat":        0.70,  # rare class — reduce false positives
    "insult":        0.50,
    "identity_hate": 0.65,  # reduce bias against identity mentions
}

comment = " ".join(sys.argv[1:])
print(f"\n=== Input Comment ===")
print(f"  \"{comment}\"")

# ── Clean ─────────────────────────────────────────────────────────────────────
data = [(1, comment)]
df = spark.createDataFrame(data, ["id", "comment_text"])
cleaned = df \
    .withColumn("comment_text", lower(col("comment_text"))) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"http\S+", "")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"\s+", " ")) \
    .withColumn("comment_text", trim(col("comment_text")))

# ── Predict ───────────────────────────────────────────────────────────────────
print("\n=== Prediction Results ===")
print(f"  {'Label':<20} {'Probability':>12} {'Threshold':>10} {'Verdict':>10}")
print(f"  {'-'*55}")

flagged = []
for label in LABELS:
    model = PipelineModel.load(f"{HDFS_BASE}/models/{label}_model")
    prediction = model.transform(cleaned)
    row = prediction.select("prediction", "probability").collect()[0]
    prob = round(float(row["probability"][1]) * 100, 2)
    threshold = THRESHOLDS[label]
    verdict = "🚨 YES" if prob/100 >= threshold else "✅ NO"
    print(f"  {label:<20} {prob:>11}% {threshold*100:>9.0f}% {verdict:>10}")
    if prob/100 >= threshold:
        flagged.append(label)

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n=== Overall Verdict ===")
if flagged:
    print(f"  🚨 TOXIC COMMENT DETECTED")
    print(f"  Categories : {', '.join(flagged)}")
else:
    print(f"  ✅ CLEAN COMMENT")

print("\n=== Done ===")
spark.stop()
