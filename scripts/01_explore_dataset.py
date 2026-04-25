from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim

spark = SparkSession.builder \
    .appName("Jigsaw Dataset Exploration") \
    .master("local[*]") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

# Load training dataset
df = spark.read.csv(
    "data/train.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

print("Schema:")
df.printSchema()

print("Total rows:", df.count())

print("Sample rows:")
df.select("id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate").show(5, truncate=80)

# Basic text cleaning
cleaned_df = df.withColumn("clean_comment", lower(col("comment_text"))) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"http\S+", "")) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"[^a-zA-Z\s]", " ")) \
    .withColumn("clean_comment", regexp_replace(col("clean_comment"), r"\s+", " ")) \
    .withColumn("clean_comment", trim(col("clean_comment")))

print("Cleaned sample:")
cleaned_df.select("comment_text", "clean_comment").show(5, truncate=100)

spark.stop()