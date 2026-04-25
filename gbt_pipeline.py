from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import time

spark = SparkSession.builder \
    .appName("HateSpeech-GBT") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
df = df.select("comment_text", col("toxic").alias("label")) \
       .filter(col("comment_text").isNotNull())

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count()} | Test: {test_df.count()}")

# ── TF-IDF stages (same as LR pipeline) ──────────────────────────────────────
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

# ── GBT Classifier ────────────────────────────────────────────────────────────
# Note: GBT requires features to be dense or uses maxBins for sparse
# We use fewer trees (20) to keep training time reasonable
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    maxDepth=5,
    stepSize=0.1
)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, gbt])

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n=== Training GBT model ===")
print("  (This will take longer than LR — ~5-10 mins)")
t0 = time.time()
model = pipeline.fit(train_df)
print(f"  Training time: {round(time.time() - t0, 2)}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== Evaluating on test set ===")
predictions = model.transform(test_df)

evaluator_auc = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)
evaluator_pr = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderPR"
)

auc = evaluator_auc.evaluate(predictions)
pr  = evaluator_pr.evaluate(predictions)
print(f"  AUC-ROC : {auc:.4f}")
print(f"  AUC-PR  : {pr:.4f}")

print("\n=== Confusion Matrix ===")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# ── Save model ────────────────────────────────────────────────────────────────
print("\n=== Saving model to HDFS ===")
model.write().overwrite().save(f"{HDFS_BASE}/models/gbt_tfidf")
print("  Saved to HDFS /models/gbt_tfidf")

print("\n=== GBT complete ===")
spark.stop()
