from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
import json, time

spark = SparkSession.builder \
    .appName("HateSpeech-FinalModel") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
df = df.select("comment_text", col("toxic").alias("label")) \
       .filter(col("comment_text").isNotNull())

# Use full training set this time — no test split
# We already know the best params from CV
total    = df.count()
n_pos    = df.filter(col("label") == 1).count()
n_neg    = df.filter(col("label") == 0).count()
ratio    = n_neg / n_pos

print(f"  Total samples : {total}")
print(f"  Class weight  : {ratio:.2f}x for toxic class")

df_weighted = df.withColumn(
    "classWeight",
    when(col("label") == 1, ratio).otherwise(1.0)
)

# ── Build final pipeline with best params from CV ─────────────────────────────
print("\n=== Building final pipeline with tuned hyperparameters ===")
print("  regParam=0.01, elasticNetParam=0.5, numFeatures=10000, minDocFreq=5")

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
    regParam=0.01,
    elasticNetParam=0.5,
    maxIter=20
)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# ── Train on full dataset ─────────────────────────────────────────────────────
print("\n=== Training final model on full dataset ===")
t0 = time.time()
final_model = pipeline.fit(df_weighted)
train_time = round(time.time() - t0, 2)
print(f"  Training time: {train_time}s")

# ── Quick sanity check on training data ──────────────────────────────────────
print("\n=== Sanity check on training data ===")
train_preds = final_model.transform(df_weighted)
evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)
train_auc = evaluator.evaluate(train_preds)
print(f"  Training AUC-ROC: {train_auc:.4f}")
print(f"  (CV test AUC was 0.8724 — training AUC should be slightly higher)")

# ── Save final model ──────────────────────────────────────────────────────────
print("\n=== Saving final model to HDFS ===")
final_model.write().overwrite().save(f"{HDFS_BASE}/models/final_model")
print("  Saved: /models/final_model")

# ── Save model metadata as JSON ───────────────────────────────────────────────
print("\n=== Saving model metadata ===")
metadata = {
    "model_type": "LogisticRegression",
    "features": "TF-IDF (HashingTF + IDF)",
    "num_features": 10000,
    "min_doc_freq": 5,
    "reg_param": 0.01,
    "elastic_net_param": 0.5,
    "max_iter": 20,
    "class_weighting": f"{ratio:.2f}x for toxic class",
    "training_samples": total,
    "training_time_sec": train_time,
    "cv_test_auc_roc": 0.8724,
    "cv_test_auc_pr": 0.4780,
    "cv_test_recall": 0.8657,
    "cv_test_precision": 0.2344,
    "cv_test_f1": 0.3689,
    "selected_over": ["LR baseline (AUC 0.87, recall 0.52)",
                      "GBT (AUC 0.84, recall 0.37)",
                      "LR weighted untuned (AUC 0.84, recall 0.69)"],
    "selection_rationale": (
        "Highest recall (0.87) minimizes missed toxic comments. "
        "Best AUC-ROC (0.8724) among class-weighted models. "
        "Low false negatives (402) critical for moderation use case."
    )
}

meta_path = "/home/aj4955_nyu_edu/hatespeech_data/model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata to {meta_path}")

# ── Print final summary ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  FINAL MODEL SUMMARY")
print("="*55)
print(f"  Algorithm     : Logistic Regression")
print(f"  Features      : TF-IDF (10,000 dims, minDocFreq=5)")
print(f"  regParam      : 0.01")
print(f"  elasticNet    : 0.5 (L1+L2 mix)")
print(f"  Class weight  : {ratio:.2f}x for toxic class")
print(f"  Training time : {train_time}s")
print(f"  ── Test Performance (from CV) ──")
print(f"  AUC-ROC       : 0.8724")
print(f"  AUC-PR        : 0.4780")
print(f"  Recall        : 0.8657")
print(f"  Precision     : 0.2344")
print(f"  F1            : 0.3689")
print(f"  True Positives: 2,592 / 2,994 toxic comments caught")
print(f"  False Negatives: 402 toxic comments missed")
print("="*55)

print("\n=== All model artifacts saved. Task 2 complete. ===")
spark.stop()
