from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover

spark = SparkSession.builder \
    .appName("Jigsaw Tokenization") \
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
    .withColumn("clean_comment", trim(col("clean_comment")))

tokenizer = Tokenizer(inputCol="clean_comment", outputCol="tokens")
tokenized_df = tokenizer.transform(cleaned_df)

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
final_df = remover.transform(tokenized_df)

final_df.select(
    "id",
    "comment_text",
    "clean_comment",
    "tokens",
    "filtered_tokens",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
).show(5, truncate=80)

spark.stop()