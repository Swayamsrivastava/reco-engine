from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.shuffle.partitions", "16")
         .appName("ml_interactions").getOrCreate())

ratings = spark.read.csv("data/raw/ratings.csv", header=True, inferSchema=True)
movies  = spark.read.csv("data/raw/movies.csv",  header=True, inferSchema=True)

# 1) remove duplicates, keep users with at least 5 ratings
ratings = ratings.dropDuplicates(["userId", "movieId"])
active = ratings.groupBy("userId").count().filter("count >= 5").select("userId")
ratings = ratings.join(active, "userId")

# 2) label: 1 if rating >= 4
ratings = ratings.withColumn("label", (F.col("rating") >= 4).cast("int"))

# 3) split by time: oldest 80% train, next 10% val, newest 10% test
cut80, cut90 = ratings.approxQuantile("timestamp", [0.8, 0.9], 0.001)
train = ratings.filter(F.col("timestamp") <  cut80)
val   = ratings.filter((F.col("timestamp") >= cut80) & (F.col("timestamp") < cut90))
test  = ratings.filter(F.col("timestamp") >= cut90)

for name, df in [("train", train), ("val", val), ("test", test)]:
    df.write.mode("overwrite").parquet(f"data/processed/{name}.parquet")
    df.agg(F.min("timestamp").alias("min_ts"), F.max("timestamp").alias("max_ts"),
           F.count("*").alias("rows")).show()

# 4) movies: genres list and release year
movies = (movies
    .withColumn("genres_list", F.split("genres", r"\|"))
    .withColumn("year", F.regexp_extract("title", r"\((\d{4})\)", 1).cast("int")))
movies.write.mode("overwrite").parquet("data/processed/movies.parquet")