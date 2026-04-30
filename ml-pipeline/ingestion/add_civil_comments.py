from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, lower, regexp_replace, trim
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import json, subprocess, csv
from collections import Counter

spark = SparkSession.builder \
    .appName("HateSpeech-RebuildCombined-Full") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE  = "hdfs:///user/aj4955_nyu_edu/hatespeech"
LOCAL_BASE = "/home/aj4955_nyu_edu/hatespeech_data/extra_datasets"

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def clean_text_df(df, col_name="comment_text"):
    return df \
        .withColumn(col_name, lower(col(col_name))) \
        .withColumn(col_name, regexp_replace(col(col_name), r"http\S+", "")) \
        .withColumn(col_name, regexp_replace(col(col_name), r"[^a-zA-Z\s]", " ")) \
        .withColumn(col_name, regexp_replace(col(col_name), r"\s+", " ")) \
        .withColumn(col_name, trim(col(col_name))) \
        .filter(col(col_name).isNotNull())

schema = StructType([
    StructField("comment_text",  StringType(),  True),
    StructField("toxic",         IntegerType(), True),
    StructField("severe_toxic",  IntegerType(), True),
    StructField("obscene",       IntegerType(), True),
    StructField("threat",        IntegerType(), True),
    StructField("insult",        IntegerType(), True),
    StructField("identity_hate", IntegerType(), True),
    StructField("source",        StringType(),  True),
])

# ── 1. Jigsaw ─────────────────────────────────────────────────────────────────
print("\n=== Loading Jigsaw ===")
jigsaw = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
jigsaw = jigsaw.select(
    "comment_text","toxic","severe_toxic","obscene",
    "threat","insult","identity_hate",
    lit("jigsaw").alias("source")
)
jigsaw = clean_text_df(jigsaw)
print(f"  Jigsaw rows: {jigsaw.count()}")

# ── 2. HateXplain ─────────────────────────────────────────────────────────────
print("\n=== Loading HateXplain ===")
with open(f"{LOCAL_BASE}/hatexplain.json") as f:
    data = json.load(f)
rows = []
for entry in data.values():
    text = " ".join(entry["post_tokens"])
    majority = Counter([a["label"] for a in entry["annotators"]]).most_common(1)[0][0]
    if majority == "hatespeech":
        rows.append((text,1,0,0,0,0,1,"hatexplain"))
    elif majority == "offensive":
        rows.append((text,1,0,0,0,1,0,"hatexplain"))
    else:
        rows.append((text,0,0,0,0,0,0,"hatexplain"))
hatexplain = spark.createDataFrame(rows, schema)
hatexplain = clean_text_df(hatexplain)
print(f"  HateXplain rows: {hatexplain.count()}")

# ── 3. Twitter ────────────────────────────────────────────────────────────────
print("\n=== Loading Twitter ===")
twitter_rows = []
with open(f"{LOCAL_BASE}/labeled_data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        text = row["tweet"]
        cls = int(row["class"])
        if cls == 0:
            twitter_rows.append((text,1,0,0,0,0,1,"twitter"))
        elif cls == 1:
            twitter_rows.append((text,1,0,1,0,1,0,"twitter"))
        else:
            twitter_rows.append((text,0,0,0,0,0,0,"twitter"))
twitter = spark.createDataFrame(twitter_rows, schema)
twitter = clean_text_df(twitter)
print(f"  Twitter rows: {twitter.count()}")

# ── 4. Civil Comments — FULL DATASET ─────────────────────────────────────────
print("\n=== Loading Civil Comments (full 1.8M) ===")
civil_raw = spark.read.csv(
    f"{HDFS_BASE}/data/civil_comments_train.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)
civil_mapped = civil_raw.select(
    col("comment_text"),
    when(col("target")          >= 0.5,1).otherwise(0).cast(IntegerType()).alias("toxic"),
    when(col("severe_toxicity") >= 0.5,1).otherwise(0).cast(IntegerType()).alias("severe_toxic"),
    when(col("obscene")         >= 0.5,1).otherwise(0).cast(IntegerType()).alias("obscene"),
    when(col("threat")          >= 0.5,1).otherwise(0).cast(IntegerType()).alias("threat"),
    when(col("insult")          >= 0.5,1).otherwise(0).cast(IntegerType()).alias("insult"),
    when(col("identity_attack") >= 0.5,1).otherwise(0).cast(IntegerType()).alias("identity_hate"),
    lit("civil_comments").alias("source")
).filter(col("comment_text").isNotNull())

# Use full dataset — no sampling
civil = clean_text_df(civil_mapped)
civil_count = civil.count()
print(f"  Civil Comments rows: {civil_count}")

print("\n  Civil Comments label distribution:")
for label in labels:
    n = civil.filter(col(label)==1).count()
    print(f"  {label:<20}: {n:>7} ({n/civil_count*100:.2f}%)")

# ── 5. Union all ─────────────────────────────────────────────────────────────
print("\n=== Combining all datasets ===")
combined = jigsaw.union(hatexplain).union(twitter).union(civil)
combined.cache()
total = combined.count()
print(f"  Total rows: {total}")

print("\n  Source breakdown:")
combined.groupBy("source").count().orderBy("source").show()

print("\n  Label distribution:")
for label in labels:
    n = combined.filter(col(label)==1).count()
    print(f"  {label:<20}: {n:>7} ({n/total*100:.2f}%)")

# ── 6. Write to temp then swap ────────────────────────────────────────────────
print("\n=== Writing combined dataset ===")
combined.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/data/combined_train_v2")
print("  Write complete.")

subprocess.run(["hadoop","fs","-rm","-r",f"{HDFS_BASE}/data/combined_train"])
subprocess.run(["hadoop","fs","-mv",
    f"{HDFS_BASE}/data/combined_train_v2",
    f"{HDFS_BASE}/data/combined_train"])
print("  Swap complete.")

# ── 7. Verify ─────────────────────────────────────────────────────────────────
print("\n=== Verifying ===")
verify = spark.read.parquet(f"{HDFS_BASE}/data/combined_train")
final_count = verify.count()
print(f"  Verified: {final_count} rows")
verify.groupBy("source").count().orderBy("source").show()

print("\n=== Civil Comments full integration complete ===")
spark.stop()
