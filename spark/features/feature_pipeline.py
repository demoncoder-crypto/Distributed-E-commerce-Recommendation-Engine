from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col

spark = SparkSession.builder.appName("RecSysFeatures").getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "user_events") \
    .load()

parsed = df.select(
    from_json(col("value").cast("string"), "user_id STRING, item_id STRING, event STRING").alias("data")
).select("data.*")

# Write to Redis (see spark/features/redis_writer.py for writer logic)
query = parsed.writeStream \
    .foreachBatch(write_to_redis) \
    .start()