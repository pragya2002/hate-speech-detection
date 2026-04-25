from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, lower, regexp_replace, trim
import sys

spark = SparkSession.builder \
    .appName("HateSpeech-Interactive") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# Get comment from command line argument
comment = " ".join(sys.argv[1:])
print(f"\n=== Input Comment ===")
print(f"  \"{comment}\"")

# ── Apply same cleaning as pipeline ──────────────────────────────────────────
data = [(1, comment)]
df = spark.createDataFrame(data, ["id", "comment_text"])

cleaned = df \
    .withColumn("comment_text", lower(col("comment_text"))) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"http\S+", "")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("comment_text", regexp_replace(col("comment_text"), r"\s+", " ")) \
    .withColumn("comment_text", trim(col("comment_text")))

# ── Load model and predict ────────────────────────────────────────────────────
model = PipelineModel.load(f"{HDFS_BASE}/models/final_model")
prediction = model.transform(cleaned)

row = prediction.select("prediction", "probability").collect()[0]
predicted = int(row["prediction"])
prob = row["probability"]
toxic_prob = round(float(prob[1]) * 100, 2)
clean_prob = round(float(prob[0]) * 100, 2)

# ── Display result ────────────────────────────────────────────────────────────
print(f"\n=== Prediction Result ===")
print(f"  Toxic probability : {toxic_prob}%")
print(f"  Clean probability : {clean_prob}%")
print(f"\n  Verdict: {'🚨 TOXIC' if predicted == 1 else '✅ NOT TOXIC'}")
print(f"\n=== Done ===")
spark.stop()
