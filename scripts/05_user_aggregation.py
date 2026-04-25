from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, sum as spark_sum, count, avg

spark = SparkSession.builder \
    .appName("User Toxicity Aggregation") \
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

# NOTE: Dataset does NOT have user_id → we simulate users
df = df.withColumn("user_id", (rand() * 1000).cast("int"))

# Aggregate per user
user_stats = df.groupBy("user_id").agg(
    count("*").alias("total_comments"),
    spark_sum("toxic").alias("toxic_comments"),
    spark_sum("severe_toxic").alias("severe_toxic_comments"),
    spark_sum("insult").alias("insult_comments"),
    spark_sum("identity_hate").alias("identity_hate_comments"),
    avg("toxic").alias("toxicity_ratio")
)

# Flag high-risk users
flagged_users = user_stats.filter(
    (col("toxicity_ratio") > 0.15) & 
    (col("total_comments") >= 50)
)

print("Top 10 users by toxicity ratio:")
user_stats.orderBy(col("toxicity_ratio").desc()).show(10)

print("Flagged high-risk users:")
flagged_users.show(10)

spark.stop()