from pyspark.sql import SparkSession
import time

spark = SparkSession.builder \
    .appName("HateSpeech-Benchmark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

results = []

# ── Benchmark 1: Raw CSV read ─────────────────────────────────────────────────
print("\n=== Benchmark 1: Raw CSV read ===")
t0 = time.time()
df_csv = spark.read.csv(
    f"{HDFS_BASE}/data/train.csv",
    header=True, inferSchema=True,
    multiLine=True, escape='"'
)
count_csv = df_csv.count()
t1 = time.time()
csv_time = round(t1 - t0, 2)
print(f"  Rows: {count_csv} | Time: {csv_time}s")
results.append(("CSV read", count_csv, csv_time))

# ── Benchmark 2: Parquet read ─────────────────────────────────────────────────
print("\n=== Benchmark 2: Partitioned Parquet read ===")
t0 = time.time()
df_parquet = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
count_parquet = df_parquet.count()
t1 = time.time()
parquet_time = round(t1 - t0, 2)
print(f"  Rows: {count_parquet} | Time: {parquet_time}s")
results.append(("Parquet read", count_parquet, parquet_time))

# ── Benchmark 3: Parquet with partition filter ────────────────────────────────
print("\n=== Benchmark 3: Parquet partition filter (toxic=1 only) ===")
t0 = time.time()
df_toxic = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned") \
    .filter("toxic = 1")
count_toxic = df_toxic.count()
t1 = time.time()
filter_time = round(t1 - t0, 2)
print(f"  Rows: {count_toxic} | Time: {filter_time}s")
results.append(("Parquet filtered", count_toxic, filter_time))

# ── Benchmark 4: CSV write ────────────────────────────────────────────────────
print("\n=== Benchmark 4: Write throughput (Parquet) ===")
t0 = time.time()
df_csv.write \
    .mode("overwrite") \
    .parquet(f"{HDFS_BASE}/data/benchmark_write_test")
t1 = time.time()
write_time = round(t1 - t0, 2)
print(f"  Write time: {write_time}s")
results.append(("Parquet write", count_csv, write_time))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Benchmark Summary ===")
print(f"  {'Operation':<25} {'Rows':>10} {'Time (s)':>10} {'Rows/sec':>12}")
print(f"  {'-'*60}")
for op, rows, t in results:
    rps = int(rows / t) if t > 0 else 0
    print(f"  {op:<25} {rows:>10} {t:>10} {rps:>12,}")

print("\n=== Cleanup benchmark write test ===")
import subprocess
subprocess.run([
    "hadoop", "fs", "-rm", "-r",
    f"{HDFS_BASE}/data/benchmark_write_test"
])

print("\n=== Benchmark complete ===")
spark.stop()
