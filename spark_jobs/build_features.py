from pyspark.sql import SparkSession, functions as F, Window

spark = (SparkSession.builder.master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "16")
         .appName("ml_features").getOrCreate())

train  = spark.read.parquet("data/processed/train.parquet")
movies = spark.read.parquet("data/processed/movies.parquet")

# ---- item features (train data only) ----
max_ts = train.agg(F.max("timestamp")).first()[0]
item = train.groupBy("movieId").agg(
    F.count("*").alias("pop_all"),
    F.avg("rating").alias("avg_rating"),
    F.sum(F.when(F.col("timestamp") > max_ts - 90 * 86400, 1).otherwise(0)).alias("pop_recent"))
gm, m = train.agg(F.avg("rating")).first()[0], 20
item = item.withColumn("avg_rating_smooth",
    (F.col("pop_all") * F.col("avg_rating") + m * gm) / (F.col("pop_all") + m))
item = item.join(movies.select("movieId", "genres_list", "year"), "movieId", "left")
item.write.mode("overwrite").parquet("data/processed/item_features.parquet")

# ---- user features ----
user = train.groupBy("userId").agg(
    F.count("*").alias("n_ratings"),
    F.avg("rating").alias("user_avg_rating"),
    F.stddev("rating").alias("user_rating_std"),
    F.max("timestamp").alias("last_ts"))
user.write.mode("overwrite").parquet("data/processed/user_features.parquet")

# ---- popularity list for cold start ----
(item.orderBy(F.desc("pop_recent"), F.desc("pop_all")).limit(500)
     .write.mode("overwrite").parquet("data/processed/popularity_global.parquet"))

# ---- user genre affinity ----
liked = (train.filter("label = 1").join(movies.select("movieId", "genres_list"), "movieId")
         .select("userId", F.explode("genres_list").alias("genre")))
aff = liked.groupBy("userId", "genre").count()
aff = aff.withColumn("share", F.col("count") / F.sum("count").over(Window.partitionBy("userId")))
aff.write.mode("overwrite").parquet("data/processed/user_genre_affinity.parquet")

# ---- last 20 liked movies per user ----
w = Window.partitionBy("userId").orderBy(F.desc("timestamp"))
hist = (train.filter("label = 1").withColumn("rk", F.row_number().over(w)).filter("rk <= 20")
        .groupBy("userId").agg(F.collect_list("movieId").alias("recent_items")))
hist.write.mode("overwrite").parquet("data/processed/user_history.parquet")
print("features done")