from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, count, when
import time

spark = SparkSession.builder \
    .appName("HateSpeech-ClassBalance") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
df = df.select("comment_text", col("toxic").alias("label")) \
       .filter(col("comment_text").isNotNull())

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ── Compute class weights ─────────────────────────────────────────────────────
print("\n=== Computing class weights ===")
total    = train_df.count()
n_pos    = train_df.filter(col("label") == 1).count()
n_neg    = train_df.filter(col("label") == 0).count()
ratio    = n_neg / n_pos

print(f"  Total samples : {total}")
print(f"  Non-toxic (0) : {n_neg} ({n_neg/total*100:.1f}%)")
print(f"  Toxic (1)     : {n_pos} ({n_pos/total*100:.1f}%)")
print(f"  Imbalance ratio: {ratio:.2f}x")
print(f"  → Weighting toxic class {ratio:.2f}x higher")

# Add weight column — toxic comments get ratio weight, non-toxic get 1.0
train_weighted = train_df.withColumn(
    "classWeight",
    when(col("label") == 1, ratio).otherwise(1.0)
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
print("\n=== Building weighted LR pipeline ===")
tokenizer = RegexTokenizer(
    inputCol="comment_text",
    outputCol="words",
    pattern="\\W"
)
remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered_words"
)
hashingTF = HashingTF(
    inputCol="filtered_words",
    outputCol="raw_features",
    numFeatures=10000
)
idf = IDF(
    inputCol="raw_features",
    outputCol="features",
    minDocFreq=5
)
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=20
)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n=== Training weighted model ===")
t0 = time.time()
model = pipeline.fit(train_weighted)
print(f"  Training time: {round(time.time() - t0, 2)}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== Evaluating on test set ===")
predictions = model.transform(test_df)

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderROC"
)
evaluator_pr = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderPR"
)

auc = evaluator_auc.evaluate(predictions)
pr  = evaluator_pr.evaluate(predictions)
print(f"  AUC-ROC : {auc:.4f}")
print(f"  AUC-PR  : {pr:.4f}")

print("\n=== Confusion Matrix ===")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# ── Compare recall improvement ────────────────────────────────────────────────
tp = predictions.filter((col("label")==1) & (col("prediction")==1.0)).count()
fn = predictions.filter((col("label")==1) & (col("prediction")==0.0)).count()
fp = predictions.filter((col("label")==0) & (col("prediction")==1.0)).count()
tn = predictions.filter((col("label")==0) & (col("prediction")==0.0)).count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== Detailed Metrics ===")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  (Baseline LR recall was ~0.52 — improvement shows class weighting worked)")

# ── Save ──────────────────────────────────────────────────────────────────────
print("\n=== Saving weighted model ===")
model.write().overwrite().save(f"{HDFS_BASE}/models/lr_weighted")
print("  Saved to HDFS /models/lr_weighted")

print("\n=== Class balancing complete ===")
spark.stop()
