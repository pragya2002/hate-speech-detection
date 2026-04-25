from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, sum as spark_sum, count, avg

spark = SparkSession.builder \
    .appName("Generate Moderation Report") \
    .getOrCreate()

df = spark.read.csv(
    "hdfs:///user/aj4955_nyu_edu/hatespeech/data/train.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

df = df.withColumn("user_id", (rand(seed=42) * 1000).cast("int"))

summary = df.agg(
    count("*").alias("total_comments"),
    spark_sum("toxic").alias("toxic_comments"),
    spark_sum("severe_toxic").alias("severe_toxic_comments"),
    spark_sum("obscene").alias("obscene_comments"),
    spark_sum("threat").alias("threat_comments"),
    spark_sum("insult").alias("insult_comments"),
    spark_sum("identity_hate").alias("identity_hate_comments")
)

user_stats = df.groupBy("user_id").agg(
    count("*").alias("total_comments"),
    spark_sum("toxic").alias("toxic_comments"),
    spark_sum("severe_toxic").alias("severe_toxic_comments"),
    spark_sum("obscene").alias("obscene_comments"),
    spark_sum("threat").alias("threat_comments"),
    spark_sum("insult").alias("insult_comments"),
    spark_sum("identity_hate").alias("identity_hate_comments"),
    avg("toxic").alias("toxicity_ratio")
)

flagged_users = user_stats.filter(
    (col("toxicity_ratio") > 0.15) &
    (col("total_comments") >= 50)
).orderBy(col("toxicity_ratio").desc())

flagged_comments = df.filter(
    (col("toxic") == 1) |
    (col("severe_toxic") == 1) |
    (col("obscene") == 1) |
    (col("threat") == 1) |
    (col("insult") == 1) |
    (col("identity_hate") == 1)
).select(
    "id",
    "user_id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
)

summary.coalesce(1).write.mode("overwrite").option("header", True).csv("hdfs:///user/aj4955_nyu_edu/hatespeech/outputs/moderation_summary")
flagged_users.coalesce(1).write.mode("overwrite").option("header", True).csv("hdfs:///user/aj4955_nyu_edu/hatespeech/outputs/flagged_users")
flagged_comments.coalesce(1).write.mode("overwrite").option("header", True).csv("hdfs:///user/aj4955_nyu_edu/hatespeech/outputs/flagged_comments")

print("Moderation report generated successfully.")
print("Files saved in:")
print("hdfs:///user/aj4955_nyu_edu/hatespeech/outputs/moderation_summary")
print("hdfs:///user/aj4955_nyu_edu/hatespeech/outputs/flagged_users")
print("hdfs:///user/aj4955_nyu_edu/hatespeech/outputs/flagged_comments")

print("Overall dataset summary:")
summary.show(truncate=False)

print("Top flagged users:")
flagged_users.show(10, truncate=False)

spark.stop()