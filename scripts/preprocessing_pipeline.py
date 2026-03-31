from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, DoubleType
)
import time

PROJECT_ID  = "recosys-489001"
BUCKET      = "recosys-data-bucket"

EDA_FILES = [
    f"gs://{BUCKET}/raw/2019-Oct.csv",
    f"gs://{BUCKET}/raw/2019-Nov.csv",
    f"gs://{BUCKET}/raw/2019-Dec.csv",
    f"gs://{BUCKET}/raw/2020-Jan.csv",
    f"gs://{BUCKET}/raw/2020-Feb.csv",
]

CHECKPOINT_DIR  = f"gs://{BUCKET}/spark_checkpoints"
PRE_KCORE_PATH  = f"gs://{BUCKET}/processed/events_pre_kcore"
CLEAN_PATH      = f"gs://{BUCKET}/processed/events_clean"
BQ_DATASET      = "recosys"
BQ_CLEAN_TABLE  = f"{PROJECT_ID}.{BQ_DATASET}.events_clean"
BQ_TEMP_BUCKET  = f"{BUCKET}/bq_temp"

PRICE_FLOOR       = 1.0
BOT_THRESHOLD_EPD = 300
CORE_K            = 3
CORE_ITERATIONS   = 10

schema = StructType([
    StructField("event_time",    StringType(),  True),
    StructField("event_type",    StringType(),  True),
    StructField("product_id",    LongType(),    True),
    StructField("category_id",   LongType(),    True),
    StructField("category_code", StringType(),  True),
    StructField("brand",         StringType(),  True),
    StructField("price",         DoubleType(),  True),
    StructField("user_id",       LongType(),    True),
    StructField("user_session",  StringType(),  True),
])

def elapsed(start):
    s = int(time.time() - start)
    return f"{s//60}m {s%60}s"


spark = SparkSession.builder \
    .appName("recosys-preprocessing") \
    .config("spark.sql.shuffle.partitions", "480") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.checkpoint.compress", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.sparkContext.setCheckpointDir(CHECKPOINT_DIR)

print("=" * 60)
print("  RecoSys Preprocessing Pipeline")
print("=" * 60)
print(f"Spark version  : {spark.version}")
print(f"Cores available: {spark.sparkContext.defaultParallelism}")
print(f"Checkpoint dir : {CHECKPOINT_DIR}")
print(f"Output path    : {CLEAN_PATH}")
print("=" * 60)

print("\n[PHASE 1] Loading raw data...")
t_phase1 = time.time()

df_raw = spark.read \
    .option("header", "true") \
    .schema(schema) \
    .csv(EDA_FILES)

df_raw = df_raw.withColumn(
    "event_time",
    F.to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss z")
)

total_raw = df_raw.count()
print(f"Raw rows loaded : {total_raw:,}")
print(f"Expected        : 288,779,227")
print(f"Match           : {total_raw == 288_779_227}")
print(f"Phase 1 done in {elapsed(t_phase1)}")


print("\n[PHASE 2] Cleaning...")
t_phase2 = time.time()

df = df_raw \
    .fillna("unknown", subset=["category_code", "brand"]) \
    .filter(F.col("user_session").isNotNull())

rows_step1 = df.count()
print(f"Step 1 (nulls)       : {rows_step1:,}  removed {total_raw - rows_step1:,}")

df = df.dropDuplicates([
    "event_time", "event_type", "product_id",
    "user_id", "user_session"
])

rows_step2 = df.count()
print(f"Step 2 (exact dedup) : {rows_step2:,}  removed {rows_step1 - rows_step2:,}")

w_dedup = Window \
    .partitionBy("user_id", "product_id", "event_type") \
    .orderBy("event_time")

df = df \
    .withColumn("prev_time", F.lag("event_time").over(w_dedup)) \
    .withColumn("gap_secs",
        F.when(F.col("prev_time").isNotNull(),
               F.col("event_time").cast("long") -
               F.col("prev_time").cast("long"))
    ) \
    .filter(
        F.col("prev_time").isNull() |
        (F.col("gap_secs") > 1)
    ) \
    .drop("prev_time", "gap_secs")

rows_step3 = df.count()
print(f"Step 3 (near dedup)  : {rows_step3:,}  removed {rows_step2 - rows_step3:,}")

df = df.filter(F.col("price") >= PRICE_FLOOR)

rows_step4 = df.count()
print(f"Step 4 (price floor) : {rows_step4:,}  removed {rows_step3 - rows_step4:,}")


bot_users = df \
    .withColumn("day", F.to_date("event_time")) \
    .groupBy("user_id", "day") \
    .agg(F.count("*").alias("daily_events")) \
    .groupBy("user_id") \
    .agg(F.avg("daily_events").alias("avg_epd")) \
    .filter(F.col("avg_epd") > BOT_THRESHOLD_EPD) \
    .select("user_id")

n_bots = bot_users.count()
df = df.join(bot_users, "user_id", "leftanti")

rows_step5 = df.count()
print(f"Step 5 (bots)        : {rows_step5:,}  removed {n_bots:,} users / {rows_step4 - rows_step5:,} events")
print(f"Phase 2 done in {elapsed(t_phase2)}")

print(f"\nSaving pre-k-core data to GCS...")
t_save = time.time()

df.write \
    .mode("overwrite") \
    .parquet(PRE_KCORE_PATH)

print(f"Saved to {PRE_KCORE_PATH}")
print(f"Save done in {elapsed(t_save)}")

df = spark.read.parquet(PRE_KCORE_PATH)
print(f"Re-loaded from GCS: {df.count():,} rows")

print("\n[PHASE 3] K-Core filtering (k=3)...")
t_phase3 = time.time()

current   = df
prev_rows = df.count()
print(f"Starting k-core with {prev_rows:,} rows\n")

for i in range(1, CORE_ITERATIONS + 1):
    t_round = time.time()

    user_counts = current \
        .groupBy("user_id") \
        .count() \
        .filter(F.col("count") >= CORE_K) \
        .select("user_id")

    item_counts = current \
        .groupBy("product_id") \
        .count() \
        .filter(F.col("count") >= CORE_K) \
        .select("product_id")

    current = current \
        .join(user_counts, "user_id",    "inner") \
        .join(item_counts, "product_id", "inner")

    current = current.checkpoint()

    cur_rows  = current.count()
    cur_users = current.select("user_id").distinct().count()
    cur_items = current.select("product_id").distinct().count()

    print(f"Round {i}: rows={cur_rows:,}  users={cur_users:,}  "
          f"items={cur_items:,}  removed={prev_rows - cur_rows:,}  "
          f"time={elapsed(t_round)}")

    if cur_rows == prev_rows:
        print(f"Converged at round {i}.")
        break

    prev_rows = cur_rows

print(f"\nK-Core final:")
print(f"  Rows  : {cur_rows:,}")
print(f"  Users : {cur_users:,}")
print(f"  Items : {cur_items:,}")
print(f"Phase 3 done in {elapsed(t_phase3)}")

print("\n[PHASE 4] Saving final cleaned data...")
t_phase4 = time.time()

current \
    .repartition(200) \
    .write \
    .mode("overwrite") \
    .parquet(CLEAN_PATH)

print(f"Parquet saved to: {CLEAN_PATH}")

current \
    .write \
    .format("bigquery") \
    .option("table", BQ_CLEAN_TABLE) \
    .option("temporaryGcsBucket", BQ_TEMP_BUCKET) \
    .mode("overwrite") \
    .save()

print(f"BigQuery table: {BQ_CLEAN_TABLE}")
print(f"Phase 4 done in {elapsed(t_phase4)}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"Raw rows           : {total_raw:,}")
print(f"After cleaning     : {rows_step5:,}")
print(f"After k-core       : {cur_rows:,}")
print(f"Total removed      : {total_raw - cur_rows:,}  "
      f"({(total_raw - cur_rows) / total_raw * 100:.1f}%)")
print(f"Users retained     : {cur_users:,}")
print(f"Items retained     : {cur_items:,}")
print(f"Output             : {CLEAN_PATH}")
print(f"BigQuery           : {BQ_CLEAN_TABLE}")
print("=" * 60)

spark.stop()