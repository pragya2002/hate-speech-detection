from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

spark = SparkSession.builder \
    .appName("Jigsaw TF-IDF Feature Extraction") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs:///user/aj4955_nyu_edu/hatespeech/data/train.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

cleaned_df = df.withColumn("clean_comment", lower(col("comment_text"))) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"http\S+", "")) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"\s+", " ")) \
    .withColumn("clean_comment", trim(col("clean_comment")))

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

idf = IDF(
    inputCol="raw_features",
    outputCol="features"
)

idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

tfidf_df.select(
    "id",
    "filtered_tokens",
    "raw_features",
    "features",
    "toxic"
).show(5, truncate=80)

print("Total rows with TF-IDF features:", tfidf_df.count())

spark.stop()