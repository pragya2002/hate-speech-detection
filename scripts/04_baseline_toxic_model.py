from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder \
    .appName("Baseline Toxic Comment Classifier") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

cleaned_df = df.withColumn("clean_comment", lower(col("comment_text"))) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"http\S+", "")) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"\s+", " ")) \
    .withColumn("clean_comment", trim(col("clean_comment"))) \
    .withColumn("label", col("toxic").cast("double"))

tokenizer = Tokenizer(inputCol="clean_comment", outputCol="tokens")
tokenized_df = tokenizer.transform(cleaned_df)

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
filtered_df = remover.transform(tokenized_df)

hashing_tf = HashingTF(
    inputCol="filtered_tokens",
    outputCol="raw_features",
    numFeatures=10000
)

featurized_df = hashing_tf.transform(filtered_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

model_df = tfidf_df.select("id", "comment_text", "features", "label")

train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=10
)

model = lr.fit(train_df)

predictions = model.transform(test_df)

predictions.select(
    "id",
    "label",
    "prediction",
    "probability"
).show(10, truncate=False)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)

print("Baseline Toxic Classifier AUC:", auc)

spark.stop()