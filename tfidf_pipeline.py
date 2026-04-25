from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, HashingTF, IDF
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("HateSpeech-TFIDF") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load partitioned parquet (faster than CSV) ────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")

# We'll do binary classification on 'toxic' label first
df = df.select("comment_text", col("toxic").alias("label")) \
       .filter(col("comment_text").isNotNull())

# Train/test split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"  Train: {train_df.count()} | Test: {test_df.count()}")

# ── Build TF-IDF Pipeline ─────────────────────────────────────────────────────
print("\n=== Building TF-IDF pipeline ===")

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
    maxIter=20
)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# ── Train ─────────────────────────────────────────────────────────────────────
print("\n=== Training model ===")
import time
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

# Confusion matrix
print("\n=== Confusion Matrix ===")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

# ── Save model ────────────────────────────────────────────────────────────────
print("\n=== Saving model to HDFS ===")
model.write().overwrite().save(f"{HDFS_BASE}/models/lr_tfidf")
print("  Saved to HDFS /models/lr_tfidf")

print("\n=== TF-IDF + Logistic Regression complete ===")
spark.stop()
