from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
import time
import json

spark = SparkSession.builder \
    .appName("HateSpeech-MultiLabel") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n=== Loading data ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/train_partitioned")
df = df.filter(col("comment_text").isNotNull())
total = df.count()
print(f"  Total rows: {total}")

# ── Labels to train ───────────────────────────────────────────────────────────
labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# ── Pipeline stages (shared across all models) ────────────────────────────────
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

# ── Train one model per label ─────────────────────────────────────────────────
results = {}

for label in labels:
    print(f"\n{'='*55}")
    print(f"  Training model for: {label}")
    print(f"{'='*55}")

    # Compute class weight for this label
    n_pos = df.filter(col(label) == 1).count()
    n_neg = df.filter(col(label) == 0).count()
    ratio = n_neg / n_pos
    print(f"  Positive samples : {n_pos} ({n_pos/total*100:.2f}%)")
    print(f"  Class weight     : {ratio:.2f}x")

    # Prepare dataframe for this label
    label_df = df.select(
        "comment_text",
        col(label).alias("label")
    ).withColumn(
        "classWeight",
        when(col("label") == 1, ratio).otherwise(1.0)
    )

    train_df, test_df = label_df.randomSplit([0.8, 0.2], seed=42)

    # Logistic Regression with best params from CV
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        regParam=0.01,
        elasticNetParam=0.5,
        maxIter=20
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    # Train
    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = round(time.time() - t0, 2)
    print(f"  Training time    : {train_time}s")

    # Evaluate
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

    tp = predictions.filter((col("label")==1) & (col("prediction")==1.0)).count()
    fn = predictions.filter((col("label")==1) & (col("prediction")==0.0)).count()
    fp = predictions.filter((col("label")==0) & (col("prediction")==1.0)).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  AUC-PR           : {pr:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  F1               : {f1:.4f}")

    # Save model
    model_path = f"{HDFS_BASE}/models/{label}_model"
    model.write().overwrite().save(model_path)
    print(f"  Saved to         : {model_path}")

    results[label] = {
        "auc_roc"      : round(auc, 4),
        "auc_pr"       : round(pr, 4),
        "recall"       : round(recall, 4),
        "precision"    : round(precision, 4),
        "f1"           : round(f1, 4),
        "n_positive"   : n_pos,
        "class_weight" : round(ratio, 2),
        "train_time_s" : train_time
    }

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  MULTI-LABEL TRAINING SUMMARY")
print(f"{'='*55}")
print(f"  {'Label':<20} {'AUC-ROC':>8} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print(f"  {'-'*55}")
for label, metrics in results.items():
    print(f"  {label:<20} {metrics['auc_roc']:>8} {metrics['recall']:>8} {metrics['precision']:>10} {metrics['f1']:>6}")

# ── Save results metadata ─────────────────────────────────────────────────────
meta_path = "/home/aj4955_nyu_edu/hatespeech_data/multilabel_metadata.json"
with open(meta_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Metadata saved to {meta_path}")

print("\n=== Multi-label training complete ===")
spark.stop()
