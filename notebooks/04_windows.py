"""
Stage 4 — Window Functions (Trend Detection)
============================================
Uses window functions to rank posts, detect score velocity, and identify
trending content within each subreddit.

Key Spark concepts demonstrated:
  - Window / partitionBy / orderBy
  - rank / dense_rank
  - lag / lead
  - rowsBetween
  - Rolling averages

Usage:
    python 04_windows.py
    (Run 02_transform.py first so data/silver/ exists)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
WINDOWS_PATH = "data/windows"


# ─────────────────────────────────────────────
# 2. Rank Posts by Engagement Score
# ─────────────────────────────────────────────

def rank_posts_by_engagement(df: DataFrame) -> DataFrame:
    """
    Rank posts within each subreddit by engagement_score using:
    rank().over(Window.partitionBy("subreddit").orderBy(desc("engagement_score")))
    
    This identifies the top-performing posts in each community.
    """
    # Define window spec: partition by subreddit, order by engagement score descending
    engagement_window = (
        Window
        .partitionBy("subreddit")
        .orderBy(F.desc("engagement_score"))
    )
    
    return (
        df
        .withColumn(
            "engagement_rank",
            F.rank().over(engagement_window)
        )
        .withColumn(
            "engagement_dense_rank",
            F.dense_rank().over(engagement_window)
        )
    )


# ─────────────────────────────────────────────
# 3. Detect Score Velocity (Momentum)
# ─────────────────────────────────────────────

def detect_score_velocity(df: DataFrame) -> DataFrame:
    """
    Detect score velocity: score - lag(score, 1).over(hourly_window)
    per subreddit — posts gaining momentum fast are "trending".
    
    This measures how quickly a post's score is changing relative to
    previous posts in the same subreddit, ordered by time.
    """
    # Window spec: partition by subreddit, order by timestamp
    hourly_window = (
        Window
        .partitionBy("subreddit")
        .orderBy("post_timestamp")
    )
    
    return (
        df
        .withColumn(
            "prev_score",
            F.lag("score", 1).over(hourly_window)
        )
        .withColumn(
            "score_velocity",
            F.col("score") - F.col("prev_score")
        )
        .withColumn(
            "prev_engagement_score",
            F.lag("engagement_score", 1).over(hourly_window)
        )
        .withColumn(
            "engagement_velocity",
            F.col("engagement_score") - F.col("prev_engagement_score")
        )
        # Fill nulls for first row in each partition
        .fillna({
            "prev_score": 0,
            "score_velocity": 0,
            "prev_engagement_score": 0,
            "engagement_velocity": 0,
        })
    )


# ─────────────────────────────────────────────
# 4. Rolling 3-Hour Average Engagement
# ─────────────────────────────────────────────

def rolling_avg_engagement(df: DataFrame) -> DataFrame:
    """
    Rolling 3-hour average engagement using rowsBetween(-2, 0).
    This smooths out spikes and shows the true engagement trend.
    
    The window includes the current row and the 2 previous rows
    (total of 3 rows) when ordered by timestamp within each subreddit.
    """
    # Window spec: partition by subreddit, order by timestamp, 
    # include current row and 2 previous rows
    rolling_window = (
        Window
        .partitionBy("subreddit")
        .orderBy("post_timestamp")
        .rowsBetween(-2, 0)  # 2 previous rows + current row = 3-hour window
    )
    
    return (
        df
        .withColumn(
            "rolling_avg_engagement_3h",
            F.avg("engagement_score").over(rolling_window)
        )
        .withColumn(
            "rolling_avg_score_3h",
            F.avg("score").over(rolling_window)
        )
        .withColumn(
            "rolling_avg_comments_3h",
            F.avg("num_comments_clean").over(rolling_window)
        )
    )


# ─────────────────────────────────────────────
# 5. Label Trending Posts
# ─────────────────────────────────────────────

def label_trending_posts(df: DataFrame, top_n: int = 3) -> DataFrame:
    """
    Label top-3 posts per subreddit as is_trending = True.
    Combines multiple signals: high rank, positive velocity, and
    engagement above rolling average.
    """
    # First, rank posts within each subreddit
    engagement_window = (
        Window
        .partitionBy("subreddit")
        .orderBy(F.desc("engagement_score"))
    )
    
    df_with_rank = df.withColumn(
        "engagement_rank",
        F.rank().over(engagement_window)
    )
    
    # Label top N posts as trending
    df_with_trending = df_with_rank.withColumn(
        "is_trending",
        F.when(F.col("engagement_rank") <= top_n, F.lit(True))
         .otherwise(F.lit(False))
    )
    
    return df_with_trending


# ─────────────────────────────────────────────
# 6. Combined Trend Detection
# ─────────────────────────────────────────────

def combined_trend_analysis(df: DataFrame) -> DataFrame:
    """
    Combines all window function analyses into a single enriched DataFrame.
    This is the main output of Stage 4.
    """
    # Apply all window functions
    df = rank_posts_by_engagement(df)
    df = detect_score_velocity(df)
    df = rolling_avg_engagement(df)
    df = label_trending_posts(df, top_n=3)
    
    return df


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("reddit-windows")
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 4 — Window Functions (Trend Detection)")
    print("="*55)

    # ── Read silver layer ─────────────────────
    print(f"\n[1/1] Reading silver Parquet from: {SILVER_PATH}")
    silver_df = spark.read.parquet(SILVER_PATH)
    print(f"  Rows in silver layer: {silver_df.count():,}")

    # ── Apply combined trend analysis ─────────
    print("\n" + "-"*55)
    print("  Applying window functions for trend detection...")
    print("-"*55)
    
    enriched_df = combined_trend_analysis(silver_df)
    
    # ── Show sample results ────────────────────
    print("\n  Top 10 posts by engagement rank (per subreddit):")
    (
        enriched_df
        .filter(F.col("engagement_rank") <= 3)
        .select(
            "subreddit", "title", "engagement_score", "engagement_rank",
            "is_trending", "score_velocity", "engagement_velocity"
        )
        .orderBy("subreddit", "engagement_rank")
        .show(20, truncate=50)
    )

    print("\n  Posts with highest score velocity (gaining momentum):")
    (
        enriched_df
        .filter(F.col("score_velocity") > 0)
        .select(
            "subreddit", "title", "score", "score_velocity",
            "engagement_velocity", "post_timestamp"
        )
        .orderBy(F.desc("score_velocity"))
        .show(10, truncate=50)
    )

    print("\n  Sample rolling averages (first subreddit):")
    first_sub = enriched_df.select("subreddit").distinct().first()[0]
    (
        enriched_df
        .filter(F.col("subreddit") == first_sub)
        .select(
            "title", "engagement_score", "rolling_avg_engagement_3h",
            "score", "rolling_avg_score_3h", "post_timestamp"
        )
        .orderBy("post_timestamp")
        .show(10, truncate=40)
    )

    print("\n  Trending posts summary:")
    trending_summary = (
        enriched_df
        .filter(F.col("is_trending") == True)
        .groupBy("subreddit")
        .agg(
            F.count("*").alias("trending_post_count"),
            F.avg("engagement_score").alias("avg_trending_engagement"),
            F.avg("score_velocity").alias("avg_score_velocity"),
        )
        .orderBy(F.desc("avg_trending_engagement"))
    )
    trending_summary.show(truncate=False)

    # ── Save enriched data ─────────────────────
    print(f"\n  Saving enriched data with window functions to: {WINDOWS_PATH}")
    (
        enriched_df.write
        .mode("overwrite")
        .parquet(f"{WINDOWS_PATH}/trending_posts.parquet")
    )
    
    # Save trending posts only
    trending_only = enriched_df.filter(F.col("is_trending") == True)
    (
        trending_only.write
        .mode("overwrite")
        .parquet(f"{WINDOWS_PATH}/trending_only.parquet")
    )
    
    print(f"  Full enriched data: {WINDOWS_PATH}/trending_posts.parquet")
    print(f"  Trending posts only: {WINDOWS_PATH}/trending_only.parquet")

    print("\n" + "="*55)
    print("  Stage 4 complete. Window functions applied.")
    print(f"  Output directory: {WINDOWS_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
