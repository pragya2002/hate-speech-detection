from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
import time, json

spark = SparkSession.builder \
    .appName("HateSpeech-MultiLabel-Combined") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

HDFS_BASE = "hdfs:///user/aj4955_nyu_edu/hatespeech"

print("\n=== Loading combined dataset ===")
df = spark.read.parquet(f"{HDFS_BASE}/data/combined_train")
df = df.filter(col("comment_text").isNotNull())
total = df.count()
print(f"  Total rows: {total}")
df.groupBy("source").count().orderBy("source").show()

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

tokenizer = RegexTokenizer(inputCol="comment_text", outputCol="words", pattern="\\W")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=5)

results = {}

for label in LABELS:
    print(f"\n{'='*55}")
    print(f"  Training: {label}")
    print(f"{'='*55}")

    n_pos = df.filter(col(label) == 1).count()
    n_neg = df.filter(col(label) == 0).count()
    ratio = n_neg / n_pos
    print(f"  Positives: {n_pos} ({n_pos/total*100:.2f}%) | Weight: {ratio:.2f}x")

    label_df = df.select(
        "comment_text",
        col(label).alias("label")
    ).withColumn("classWeight", when(col("label") == 1, ratio).otherwise(1.0))

    train_df, test_df = label_df.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(
        featuresCol="features", labelCol="label", weightCol="classWeight",
        regParam=0.01, elasticNetParam=0.5, maxIter=20
    )

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = round(time.time() - t0, 2)

    predictions = model.transform(test_df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

    auc = evaluator.evaluate(predictions)
    pr  = pr_eval.evaluate(predictions)

    tp = predictions.filter((col("label")==1) & (col("prediction")==1.0)).count()
    fn = predictions.filter((col("label")==1) & (col("prediction")==0.0)).count()
    fp = predictions.filter((col("label")==0) & (col("prediction")==1.0)).count()

    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall    = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0

    print(f"  Time: {train_time}s | AUC-ROC: {auc:.4f} | AUC-PR: {pr:.4f}")
    print(f"  Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")

    model.write().overwrite().save(f"{HDFS_BASE}/models/{label}_model")
    print(f"  Saved to HDFS /models/{label}_model")

    results[label] = {
        "auc_roc": round(auc,4), "auc_pr": round(pr,4),
        "recall": round(recall,4), "precision": round(precision,4),
        "f1": round(f1,4), "n_positive": n_pos,
        "training_data": "jigsaw+hatexplain+twitter"
    }

print(f"\n{'='*55}")
print("  SUMMARY")
print(f"{'='*55}")
print(f"  {'Label':<20} {'AUC-ROC':>8} {'Recall':>8} {'Precision':>10} {'F1':>6}")
print(f"  {'-'*55}")
for label, m in results.items():
    print(f"  {label:<20} {m['auc_roc']:>8} {m['recall']:>8} {m['precision']:>10} {m['f1']:>6}")

with open("/home/aj4955_nyu_edu/hatespeech_data/multilabel_combined_metadata.json","w") as f:
    json.dump(results, f, indent=2)

print("\n=== Combined training complete ===")
spark.stop()
