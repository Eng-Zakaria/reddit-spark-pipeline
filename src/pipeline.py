"""
Main Pipeline Orchestration Script
===================================
Runs all stages of the Reddit Spark pipeline in sequence.
This script orchestrates the entire ETL process from ingestion to output.

Usage:
    python -m src.pipeline
    or
    python src/pipeline.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession


# ─────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────

SUBREDDITS = [
    "technology",
    "datascience",
    "programming",
    "worldnews",
    "gaming",
    "MachineLearning",
    "science",
    "business",
    "movies",
    "todayilearned",
]

POSTS_PER_SUB = 100


# ─────────────────────────────────────────────
# Spark Session Factory
# ─────────────────────────────────────────────

def create_spark_session(app_name: str = "reddit-pipeline") -> SparkSession:
    """
    Create a SparkSession with optimized configuration for local execution.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        .getOrCreate()
    )


# ─────────────────────────────────────────────
# Stage Execution Functions
# ─────────────────────────────────────────────

def run_stage_1_ingestion(spark: SparkSession):
    """
    Stage 1: Ingest - Fetch data from Reddit API and write bronze layer.
    """
    print("\n" + "="*60)
    print("  STAGE 1: INGESTION")
    print("="*60)
    
    from notebooks.ingestion import fetch_all_subreddits, POST_SCHEMA, BRONZE_PATH
    import requests
    import time
    
    # Fetch data
    print("\nFetching posts from Reddit API...")
    HEADERS = {"User-Agent": "spark-learning-project/1.0 (portfolio project)"}
    
    def fetch_subreddit(subreddit: str, limit: int = 100):
        url = f"https://www.reddit.com/r/{subreddit}/top.json"
        params = {"limit": limit, "t": "day"}
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Could not fetch r/{subreddit}: {e}")
            return []
        
        children = response.json().get("data", {}).get("children", [])
        records = []
        for child in children:
            d = child.get("data", {})
            records.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "author": d.get("author"),
                "subreddit": subreddit.lower(),
                "score": d.get("score"),
                "num_comments": d.get("num_comments"),
                "upvote_ratio": d.get("upvote_ratio"),
                "created_utc": int(d.get("created_utc", 0)),
                "url": d.get("url"),
                "permalink": d.get("permalink"),
                "is_self": d.get("is_self"),
                "selftext": d.get("selftext"),
                "link_flair_text": d.get("link_flair_text"),
                "over_18": d.get("over_18"),
                "is_video": d.get("is_video"),
                "domain": d.get("domain"),
            })
        return records
    
    all_records = []
    for sub in SUBREDDITS:
        print(f"  Fetching r/{sub}...")
        records = fetch_subreddit(sub, POSTS_PER_SUB)
        all_records.extend(records)
        print(f"    → {len(records)} posts fetched")
        time.sleep(1)
    
    print(f"\nTotal records fetched: {len(all_records)}")
    
    # Create DataFrame
    from src.schema import POST_SCHEMA
    df = spark.createDataFrame(all_records, schema=POST_SCHEMA)
    
    # Write bronze layer
    BRONZE_PATH = "data/bronze/reddit_posts.parquet"
    print(f"\nWriting bronze Parquet to: {BRONZE_PATH}")
    df.write.mode("overwrite").parquet(BRONZE_PATH)
    print("Stage 1 complete.\n")
    
    return df


def run_stage_2_transform(spark: SparkSession):
    """
    Stage 2: Transform - Clean and enrich data, write silver layer.
    """
    print("\n" + "="*60)
    print("  STAGE 2: TRANSFORM")
    print("="*60)
    
    from src.schema import SILVER_COLUMNS
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType
    
    BRONZE_PATH = "data/bronze/reddit_posts.parquet"
    SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
    
    # Read bronze
    print(f"\nReading bronze from: {BRONZE_PATH}")
    df = spark.read.parquet(BRONZE_PATH)
    print(f"Rows: {df.count():,}")
    
    # Quality filters
    COMMENT_WEIGHT = 3
    MIN_SCORE = 1
    MIN_TITLE_LENGTH = 5
    BRIGADING_RATIO = 0.5
    
    print("\nApplying quality filters...")
    df = (
        df
        .filter(F.col("title").isNotNull())
        .filter(F.length(F.trim(F.col("title"))) >= MIN_TITLE_LENGTH)
        .filter(F.col("score").isNotNull())
        .filter(F.col("score") >= MIN_SCORE)
        .filter(F.col("author").isNotNull())
        .filter(~F.col("author").isin("[deleted]", "[removed]", "AutoModerator"))
        .filter(F.col("created_utc").isNotNull())
        .filter(F.col("created_utc") > 0)
        .dropDuplicates(["id"])
    )
    
    # Add time columns
    print("Adding time-derived columns...")
    df = (
        df
        .withColumn("post_timestamp", F.to_timestamp(F.from_unixtime(F.col("created_utc"))))
        .withColumn("hour_of_day", F.hour("post_timestamp"))
        .withColumn("day_of_week_num", F.dayofweek("post_timestamp"))
        .withColumn("day_of_week", F.date_format("post_timestamp", "EEEE"))
        .withColumn("post_date", F.to_date("post_timestamp"))
        .withColumn("post_age_hours", (F.unix_timestamp(F.current_timestamp()) - F.col("created_utc")).cast("double") / 3600)
    )
    
    # Add engagement columns
    print("Computing engagement score...")
    df = (
        df
        .withColumn("num_comments_clean", F.coalesce(F.col("num_comments"), F.lit(0)).cast(IntegerType()))
        .withColumn("engagement_score", F.col("score") + (F.col("num_comments_clean") * F.lit(COMMENT_WEIGHT)))
        .withColumn("comment_ratio", F.when(F.col("score") > 0, F.col("num_comments_clean").cast("double") / F.col("score")).otherwise(F.lit(0.0)))
        .withColumn("engagement_tier", F.when(F.col("engagement_score") >= 50_000, F.lit("viral")).when(F.col("engagement_score") >= 10_000, F.lit("high")).when(F.col("engagement_score") >= 1_000, F.lit("medium")).otherwise(F.lit("low")))
    )
    
    # Add flags
    print("Adding quality flags...")
    df = (
        df
        .withColumn("is_brigaded", F.when((F.col("upvote_ratio") < BRIGADING_RATIO) & (F.col("score") > 100), F.lit(True)).otherwise(F.lit(False)))
        .withColumn("is_link_post", F.when(F.col("is_self") == False, F.lit(True)).otherwise(F.lit(False)))
        .withColumn("is_controversial", F.when(F.col("comment_ratio") > 1.0, F.lit(True)).otherwise(F.lit(False)))
        .withColumn("is_external_link", F.when(~F.col("domain").isin("self", "i.redd.it", "v.redd.it", "reddit.com", "i.imgur.com", "imgur.com") & F.col("domain").isNotNull(), F.lit(True)).otherwise(F.lit(False)))
    )
    
    # Clean text
    print("Cleaning text columns...")
    df = (
        df
        .withColumn("subreddit", F.lower(F.trim(F.col("subreddit"))))
        .withColumn("author", F.trim(F.col("author")))
        .withColumn("selftext_clean", F.regexp_replace(F.coalesce(F.col("selftext"), F.lit("")), r"[\r\n\t]+", " "))
        .withColumn("title_word_count", F.size(F.split(F.trim(F.col("title")), r"\s+")))
        .fillna({"link_flair_text": "none", "domain": "unknown"})
    )
    
    # Select final columns
    df = df.select(SILVER_COLUMNS)
    
    # Write silver
    print(f"\nWriting silver to: {SILVER_PATH}")
    df.write.mode("overwrite").parquet(SILVER_PATH)
    print("Stage 2 complete.\n")
    
    return df


def run_stage_3_aggregations(spark: SparkSession):
    """
    Stage 3: Aggregations - Generate leaderboards and pivot tables.
    """
    print("\n" + "="*60)
    print("  STAGE 3: AGGREGATIONS")
    print("="*60)
    
    from pyspark.sql import functions as F
    from src.udfs import slice_array
    
    SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
    AGGREGATIONS_PATH = "data/aggregations"
    
    print(f"\nReading silver from: {SILVER_PATH}")
    df = spark.read.parquet(SILVER_PATH)
    print(f"Rows: {df.count():,}")
    
    # Subreddit leaderboard
    print("\nGenerating subreddit leaderboard...")
    leaderboard = (
        df.groupBy("subreddit")
        .agg(
            F.avg("engagement_score").alias("avg_engagement_score"),
            F.sum("engagement_score").alias("total_engagement_score"),
            F.count("*").alias("total_posts"),
            F.max("engagement_score").alias("max_engagement_score"),
        )
        .orderBy(F.desc("avg_engagement_score"))
    )
    leaderboard.write.mode("overwrite").parquet(f"{AGGREGATIONS_PATH}/subreddit_leaderboard.parquet")
    
    # Hourly volume
    print("Generating hourly post volume...")
    hourly = (
        df.groupBy("subreddit", "hour_of_day")
        .agg(F.count("*").alias("post_count"))
        .orderBy("subreddit", "hour_of_day")
    )
    hourly.write.mode("overwrite").parquet(f"{AGGREGATIONS_PATH}/hourly_volume.parquet")
    
    # Pivot table
    print("Generating engagement tier pivot...")
    pivot_df = df.groupBy("engagement_tier").pivot("subreddit").agg(F.count("*"))
    pivot_df.write.mode("overwrite").parquet(f"{AGGREGATIONS_PATH}/engagement_tier_pivot.parquet")
    
    print("Stage 3 complete.\n")


def run_stage_4_windows(spark: SparkSession):
    """
    Stage 4: Window Functions - Trend detection and ranking.
    """
    print("\n" + "="*60)
    print("  STAGE 4: WINDOW FUNCTIONS")
    print("="*60)
    
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    
    SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
    WINDOWS_PATH = "data/windows"
    
    print(f"\nReading silver from: {SILVER_PATH}")
    df = spark.read.parquet(SILVER_PATH)
    
    # Rank posts
    print("Ranking posts by engagement...")
    engagement_window = Window.partitionBy("subreddit").orderBy(F.desc("engagement_score"))
    df = df.withColumn("engagement_rank", F.rank().over(engagement_window))
    
    # Score velocity
    print("Calculating score velocity...")
    hourly_window = Window.partitionBy("subreddit").orderBy("post_timestamp")
    df = (
        df.withColumn("prev_score", F.lag("score", 1).over(hourly_window))
        .withColumn("score_velocity", F.col("score") - F.col("prev_score"))
        .fillna({"prev_score": 0, "score_velocity": 0})
    )
    
    # Rolling average
    print("Calculating rolling averages...")
    rolling_window = Window.partitionBy("subreddit").orderBy("post_timestamp").rowsBetween(-2, 0)
    df = df.withColumn("rolling_avg_engagement_3h", F.avg("engagement_score").over(rolling_window))
    
    # Label trending
    print("Labeling trending posts...")
    df = df.withColumn("is_trending", F.when(F.col("engagement_rank") <= 3, F.lit(True)).otherwise(F.lit(False)))
    
    # Save
    df.write.mode("overwrite").parquet(f"{WINDOWS_PATH}/trending_posts.parquet")
    print("Stage 4 complete.\n")


def run_stage_5_enrichment(spark: SparkSession):
    """
    Stage 5: Enrichment - Topic tagging and category joins.
    """
    print("\n" + "="*60)
    print("  STAGE 5: ENRICHMENT")
    print("="*60)
    
    from pyspark.sql import functions as F
    from src.udfs import detect_topic
    
    SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
    ENRICHMENT_PATH = "data/enrichment"
    
    print(f"\nReading silver from: {SILVER_PATH}")
    df = spark.read.parquet(SILVER_PATH)
    
    # Apply topic tagging
    print("Applying topic tagging UDF...")
    df = df.withColumn("topic", detect_topic(F.col("title")))
    
    # Create category lookup
    print("Creating category lookup...")
    category_data = [
        ("technology", "tech"), ("datascience", "tech"), ("programming", "tech"),
        ("machinelearning", "tech"), ("worldnews", "news"), ("business", "news"),
        ("gaming", "gaming"), ("science", "science"), ("movies", "entertainment"),
        ("todayilearned", "general"),
    ]
    lookup_df = spark.createDataFrame(category_data, ["subreddit", "category"])
    
    # Broadcast join
    print("Applying broadcast join...")
    df = df.join(F.broadcast(lookup_df), on="subreddit", how="left")
    
    # Save
    df.write.mode("overwrite").parquet(f"{ENRICHMENT_PATH}/enriched_posts.parquet")
    print("Stage 5 complete.\n")


def run_stage_6_optimization(spark: SparkSession):
    """
    Stage 6: Optimization - Caching and partitioning.
    """
    print("\n" + "="*60)
    print("  STAGE 6: OPTIMIZATION")
    print("="*60)
    
    SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
    OPTIMIZATION_PATH = "data/optimization"
    
    print(f"\nReading silver from: {SILVER_PATH}")
    df = spark.read.parquet(SILVER_PATH)
    
    # Repartition
    print("Repartitioning by subreddit...")
    num_partitions = df.select("subreddit").distinct().count()
    df = df.repartition(num_partitions, "subreddit")
    print(f"Repartitioned to {num_partitions} partitions")
    
    # Save
    df.write.mode("overwrite").parquet(f"{OPTIMIZATION_PATH}/repartitioned_by_subreddit.parquet")
    print("Stage 6 complete.\n")


def run_stage_7_output(spark: SparkSession):
    """
    Stage 7: Output - Gold layer and visualizations.
    """
    print("\n" + "="*60)
    print("  STAGE 7: OUTPUT")
    print("="*60)
    
    from pyspark.sql import functions as F
    
    SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
    GOLD_PATH = "data/gold"
    
    print(f"\nReading silver from: {SILVER_PATH}")
    df = spark.read.parquet(SILVER_PATH)
    
    # Write gold layer
    print("Writing gold layer partitioned by subreddit and day_of_week...")
    df.write.mode("overwrite").partitionBy("subreddit", "day_of_week").parquet(GOLD_PATH)
    
    print("Stage 7 complete.\n")


# ─────────────────────────────────────────────
# Main Pipeline Execution
# ─────────────────────────────────────────────

def main():
    """
    Run the complete pipeline from ingestion to output.
    """
    print("\n" + "="*60)
    print("  REDDIT SPARK PIPELINE - FULL EXECUTION")
    print("="*60)
    
    spark = create_spark_session("reddit-pipeline-full")
    spark.sparkContext.setLogLevel("WARN")
    
    try:
        # Run all stages
        run_stage_1_ingestion(spark)
        run_stage_2_transform(spark)
        run_stage_3_aggregations(spark)
        run_stage_4_windows(spark)
        run_stage_5_enrichment(spark)
        run_stage_6_optimization(spark)
        run_stage_7_output(spark)
        
        print("\n" + "="*60)
        print("  PIPELINE COMPLETE - ALL STAGES EXECUTED SUCCESSFULLY")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
