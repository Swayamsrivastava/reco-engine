from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
df = spark.read.csv("data/raw/ratings.csv", header=True, inferSchema=True)
print("rows:", df.count())
df.show(5)