from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when
import time

spark = SparkSession.builder \
    .appName("HateSpeech-CrossVal") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
df = df.select("comment_text", col("toxic").alias("label")) \
       .filter(col("comment_text").isNotNull())

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Add class weights
total = train_df.count()
n_pos = train_df.filter(col("label") == 1).count()
n_neg = train_df.filter(col("label") == 0).count()
ratio = n_neg / n_pos

train_weighted = train_df.withColumn(
    "classWeight",
    when(col("label") == 1, ratio).otherwise(1.0)
)
print(f"  Train: {total} | Test: {test_df.count()}")
print(f"  Class weight applied: {ratio:.2f}x for toxic class")

# ── Pipeline stages ───────────────────────────────────────────────────────────
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

# ── Param grid ────────────────────────────────────────────────────────────────
# Keep grid small — 3 values x 2 values = 6 combos x 3 folds = 18 fits
print("\n=== Building param grid (6 combinations x 3 folds = 18 fits) ===")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
    .build()

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2  # run 2 folds in parallel
)

# ── Run cross-validation ──────────────────────────────────────────────────────
print("\n=== Running 3-fold cross-validation ===")
print("  (This will take ~10-15 mins — 18 model fits total)")
t0 = time.time()
cv_model = cv.fit(train_weighted)
elapsed = round(time.time() - t0, 2)
print(f"  CV time: {elapsed}s")

# ── Results ───────────────────────────────────────────────────────────────────
print("\n=== Cross-validation results ===")
print(f"  {'regParam':>10} {'elasticNet':>12} {'AUC-ROC':>10}")
print(f"  {'-'*35}")
for params, score in zip(paramGrid, cv_model.avgMetrics):
    reg = params[lr.regParam]
    en  = params[lr.elasticNetParam]
    print(f"  {reg:>10} {en:>12} {score:>10.4f}")

best_score = max(cv_model.avgMetrics)
best_idx   = cv_model.avgMetrics.index(best_score)
best_params = paramGrid[best_idx]
print(f"\n  Best AUC-ROC : {best_score:.4f}")
print(f"  Best regParam: {best_params[lr.regParam]}")
print(f"  Best elasticNetParam: {best_params[lr.elasticNetParam]}")

# ── Evaluate best model on held-out test set ──────────────────────────────────
print("\n=== Evaluating best model on test set ===")
predictions = cv_model.transform(test_df)

auc = evaluator.evaluate(predictions)
pr_eval = BinaryClassificationEvaluator(
    labelCol="label", metricName="areaUnderPR"
)
pr = pr_eval.evaluate(predictions)
print(f"  AUC-ROC : {auc:.4f}")
print(f"  AUC-PR  : {pr:.4f}")

print("\n=== Confusion Matrix ===")
predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

tp = predictions.filter((col("label")==1) & (col("prediction")==1.0)).count()
fn = predictions.filter((col("label")==1) & (col("prediction")==0.0)).count()
fp = predictions.filter((col("label")==0) & (col("prediction")==1.0)).count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== Final Metrics ===")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")

# ── Save best model ───────────────────────────────────────────────────────────
print("\n=== Saving best model to HDFS ===")
cv_model.bestModel.write().overwrite().save(f"{HDFS_BASE}/models/lr_best")
print("  Saved to HDFS /models/lr_best")

print("\n=== Cross-validation complete ===")
spark.stop()
