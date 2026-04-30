import json
import csv
import os
from collections import Counter
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = SparkSession.builder \
    .appName("HateSpeech-Harmonize") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE  = "hdfs:///user/aj4955_nyu_edu/hatespeech"
LOCAL_BASE = "/home/aj4955_nyu_edu/hatespeech_data/extra_datasets"

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Schema ────────────────────────────────────────────────────────────────────
schema = StructType([
    StructField("comment_text",   StringType(),  True),
    StructField("toxic",          IntegerType(), True),
    StructField("severe_toxic",   IntegerType(), True),
    StructField("obscene",        IntegerType(), True),
    StructField("threat",         IntegerType(), True),
    StructField("insult",         IntegerType(), True),
    StructField("identity_hate",  IntegerType(), True),
    StructField("source",         StringType(),  True),
])

# ── 1. Load Jigsaw (already on HDFS) ─────────────────────────────────────────
print("\n=== Loading Jigsaw dataset ===")
jigsaw = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
jigsaw = jigsaw.select(
    col("comment_text"),
    col("toxic"),
    col("severe_toxic"),
    col("obscene"),
    col("threat"),
    col("insult"),
    col("identity_hate"),
).withColumn("source", col("toxic").cast("string")) \
 .drop("source") \
 .withColumn("source", col("toxic").cast("string"))

# Rebuild with source column properly
from pyspark.sql.functions import lit
jigsaw = jigsaw.select(
    "comment_text","toxic","severe_toxic","obscene",
    "threat","insult","identity_hate",
    lit("jigsaw").alias("source")
)
print(f"  Jigsaw rows: {jigsaw.count()}")

# ── 2. Process HateXplain ─────────────────────────────────────────────────────
print("\n=== Processing HateXplain dataset ===")
with open(f"{LOCAL_BASE}/hatexplain.json") as f:
    data = json.load(f)

hatexplain_rows = []
for post_id, entry in data.items():
    # Join tokens into text
    text = clean_text(" ".join(entry["post_tokens"]))
    if not text.strip():
        continue

    # Majority vote across 3 annotators
    labels_list = [a["label"] for a in entry["annotators"]]
    majority = Counter(labels_list).most_common(1)[0][0]

    # Map to our 6 labels
    if majority == "hatespeech":
        row = (text, 1, 0, 0, 0, 0, 1, "hatexplain")  # toxic + identity_hate
    elif majority == "offensive":
        row = (text, 1, 0, 0, 0, 1, 0, "hatexplain")  # toxic + insult
    else:  # normal
        row = (text, 0, 0, 0, 0, 0, 0, "hatexplain")

    hatexplain_rows.append(row)

hatexplain_df = spark.createDataFrame(hatexplain_rows, schema)
print(f"  HateXplain rows  : {hatexplain_df.count()}")
print(f"  Hate speech posts: {hatexplain_df.filter(col('identity_hate')==1).count()}")
print(f"  Offensive posts  : {hatexplain_df.filter(col('insult')==1).count()}")
print(f"  Normal posts     : {hatexplain_df.filter(col('toxic')==0).count()}")

# ── 3. Process Twitter hate speech ───────────────────────────────────────────
print("\n=== Processing Twitter hate speech dataset ===")
twitter_rows = []
with open(f"{LOCAL_BASE}/labeled_data.csv", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = clean_text(row["tweet"])
        if not text.strip():
            continue
        cls = int(row["class"])
        if cls == 0:    # hate speech
            r = (text, 1, 0, 0, 0, 0, 1, "twitter")
        elif cls == 1:  # offensive
            r = (text, 1, 0, 1, 0, 1, 0, "twitter")  # toxic + obscene + insult
        else:           # neither
            r = (text, 0, 0, 0, 0, 0, 0, "twitter")
        twitter_rows.append(r)

twitter_df = spark.createDataFrame(twitter_rows, schema)
print(f"  Twitter rows     : {twitter_df.count()}")
print(f"  Hate speech      : {twitter_df.filter(col('identity_hate')==1).count()}")
print(f"  Offensive        : {twitter_df.filter(col('insult')==1).count()}")
print(f"  Neither          : {twitter_df.filter(col('toxic')==0).count()}")

# ── 4. Combine all datasets ───────────────────────────────────────────────────
print("\n=== Combining all datasets ===")
combined = jigsaw.union(hatexplain_df).union(twitter_df)
total = combined.count()
print(f"  Total combined rows: {total}")
print(f"\n  Source breakdown:")
combined.groupBy("source").count().orderBy("source").show()

print(f"\n  Label distribution:")
for label in LABELS:
    n = combined.filter(col(label)==1).count()
    print(f"  {label:<20}: {n:>7} ({n/total*100:.2f}%)")

# ── 5. Save combined dataset to HDFS ─────────────────────────────────────────
print("\n=== Saving combined dataset to HDFS ===")
combined.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/data/combined_train")

# Verify
verify = spark.read.parquet(f"{HDFS_BASE}/data/combined_train")
print(f"  Verified: {verify.count()} rows saved")
verify.printSchema()

print("\n=== Harmonization complete ===")
spark.stop()
