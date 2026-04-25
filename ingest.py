from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

spark = SparkSession.builder \
    .appName("HateSpeech-Ingest") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load ──────────────────────────────────────────────────────────────────────
print("\n=== Loading train.csv ===")
train = spark.read.csv(
    f"{HDFS_BASE}/data/train.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

print("\n=== Schema ===")
train.printSchema()

print(f"\n=== Row count: {train.count()} ===")

# ── Null check ────────────────────────────────────────────────────────────────
print("\n=== Null counts per column ===")
from pyspark.sql.functions import count, isnan, isnull
null_counts = train.select([
    count(when(isnull(c), c)).alias(c) for c in train.columns
])
null_counts.show()

# ── Label distribution ────────────────────────────────────────────────────────
label_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
print("\n=== Label distribution (1 = flagged) ===")
for lbl in label_cols:
    flagged = train.filter(col(lbl) == 1).count()
    pct = flagged / train.count() * 100
    print(f"  {lbl:20s}: {flagged:6d} ({pct:.2f}%)")

# ── Partition by toxic label and write back to HDFS ───────────────────────────
print("\n=== Writing partitioned dataset to HDFS ===")
train.write \
    .partitionBy("toxic") \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/data/train_partitioned")

print("\n=== Verifying partitioned output ===")
partitioned = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
print(f"Reloaded row count: {partitioned.count()}")
partitioned.printSchema()

print("\n Ingestion complete.")
spark.stop()
