"""
Stage 3 — Aggregations (Subreddit & Hourly Analytics)
======================================================
Reads the silver layer and performs various aggregations to generate
leaderboards, hourly patterns, and pivot tables for visualization.

Key Spark concepts demonstrated:
  - groupBy / agg
  - avg / sum / count / max
  - pivot
  - collect_list
  - slice UDF

Usage:
    python 03_aggregations.py
    (Run 02_transform.py first so data/silver/ exists)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.udf import udf


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
AGGREGATIONS_PATH = "data/aggregations"


# ─────────────────────────────────────────────
# 2. Subreddit Leaderboard
# ─────────────────────────────────────────────

def subreddit_leaderboard(df: DataFrame) -> DataFrame:
    """
    Average engagement score, total posts, top score per subreddit.
    This creates a leaderboard showing which subreddits are most engaging.
    """
    return (
        df
        .groupBy("subreddit")
        .agg(
            F.avg("engagement_score").alias("avg_engagement_score"),
            F.sum("engagement_score").alias("total_engagement_score"),
            F.count("*").alias("total_posts"),
            F.max("engagement_score").alias("max_engagement_score"),
            F.max("score").alias("max_score"),
            F.avg("num_comments_clean").alias("avg_comments"),
        )
        .orderBy(F.desc("avg_engagement_score"))
    )


# ─────────────────────────────────────────────
# 3. Hourly Post Volume
# ─────────────────────────────────────────────

def hourly_post_volume(df: DataFrame) -> DataFrame:
    """
    Hourly post volume per subreddit: when do people post?
    Uses groupBy("subreddit", "hour_of_day") to find peak posting times.
    """
    return (
        df
        .groupBy("subreddit", "hour_of_day")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("engagement_score").alias("avg_engagement"),
            F.sum("engagement_score").alias("total_engagement"),
        )
        .orderBy("subreddit", "hour_of_day")
    )


# ─────────────────────────────────────────────
# 4. Pivot Table: Subreddits vs Engagement Tiers
# ─────────────────────────────────────────────

def engagement_tier_pivot(df: DataFrame) -> DataFrame:
    """
    Pivot table: subreddits as columns, engagement_tier as rows.
    Great for visualization in README - shows distribution of post quality.
    """
    return (
        df
        .groupBy("engagement_tier")
        .pivot("subreddit")
        .agg(F.count("*"))
        .orderBy(
            F.when(F.col("engagement_tier") == "viral", 1)
             .when(F.col("engagement_tier") == "high", 2)
             .when(F.col("engagement_tier") == "medium", 3)
             .otherwise(4)
        )
    )


# ─────────────────────────────────────────────
# 5. Collect Top Post Titles per Subreddit
# ─────────────────────────────────────────────

# UDF to slice array and get first N elements
@udf(returnType=ArrayType(StringType()))
def slice_array(arr, n):
    """Return first n elements of array, or all if array is shorter."""
    if arr is None:
        return []
    return arr[:n]


def top_titles_per_subreddit(df: DataFrame, top_n: int = 5) -> DataFrame:
    """
    Collect top 5 post titles per subreddit using collect_list + slice UDF.
    Returns the most engaging posts from each community.
    """
    return (
        df
        .groupBy("subreddit")
        .agg(
            F.collect_list(
                F.struct(
                    F.col("title"),
                    F.col("engagement_score"),
                    F.col("score"),
                    F.col("num_comments_clean")
                )
            )
            .alias("posts_struct")
        )
        .withColumn(
            "top_posts",
            slice_array(F.col("posts_struct"), F.lit(top_n))
        )
        .select("subreddit", "top_posts")
        .orderBy("subreddit")
    )


# ─────────────────────────────────────────────
# 6. Engagement Tier Distribution
# ─────────────────────────────────────────────

def engagement_tier_distribution(df: DataFrame) -> DataFrame:
    """
    Overall distribution of engagement tiers across all posts.
    Useful for understanding the quality distribution of your dataset.
    """
    return (
        df
        .groupBy("engagement_tier")
        .agg(
            F.count("*").alias("post_count"),
            F.round(F.count("*") / F.count("*") * 100, 2).alias("percentage"),
            F.avg("engagement_score").alias("avg_engagement"),
            F.min("engagement_score").alias("min_engagement"),
            F.max("engagement_score").alias("max_engagement"),
        )
        .orderBy(
            F.when(F.col("engagement_tier") == "viral", 1)
             .when(F.col("engagement_tier") == "high", 2)
             .when(F.col("engagement_tier") == "medium", 3)
             .otherwise(4)
        )
    )


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("reddit-aggregations")
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 3 — Aggregations")
    print("="*55)

    # ── Read silver layer ─────────────────────
    print(f"\n[1/1] Reading silver Parquet from: {SILVER_PATH}")
    silver_df = spark.read.parquet(SILVER_PATH)
    print(f"  Rows in silver layer: {silver_df.count():,}")

    # ── Subreddit Leaderboard ─────────────────
    print("\n" + "-"*55)
    print("  Subreddit Leaderboard (by avg engagement score)")
    print("-"*55)
    leaderboard = subreddit_leaderboard(silver_df)
    leaderboard.show(truncate=False)

    # Save leaderboard
    leaderboard_path = f"{AGGREGATIONS_PATH}/subreddit_leaderboard.parquet"
    (
        leaderboard.write
        .mode("overwrite")
        .parquet(leaderboard_path)
    )
    print(f"\n  Leaderboard saved to: {leaderboard_path}")

    # ── Hourly Post Volume ────────────────────
    print("\n" + "-"*55)
    print("  Hourly Post Volume per Subreddit")
    print("-"*55)
    hourly = hourly_post_volume(silver_df)
    hourly.show(24, truncate=False)  # Show all 24 hours for first subreddit

    # Save hourly volume
    hourly_path = f"{AGGREGATIONS_PATH}/hourly_volume.parquet"
    (
        hourly.write
        .mode("overwrite")
        .parquet(hourly_path)
    )
    print(f"\n  Hourly volume saved to: {hourly_path}")

    # ── Engagement Tier Pivot ─────────────────
    print("\n" + "-"*55)
    print("  Engagement Tier Pivot Table")
    print("-"*55)
    pivot_df = engagement_tier_pivot(silver_df)
    pivot_df.show(truncate=False)

    # Save pivot table
    pivot_path = f"{AGGREGATIONS_PATH}/engagement_tier_pivot.parquet"
    (
        pivot_df.write
        .mode("overwrite")
        .parquet(pivot_path)
    )
    print(f"\n  Pivot table saved to: {pivot_path}")

    # ── Top Titles per Subreddit ───────────────
    print("\n" + "-"*55)
    print("  Top 5 Post Titles per Subreddit")
    print("-"*55)
    top_titles = top_titles_per_subreddit(silver_df, top_n=5)
    top_titles.show(truncate=False)

    # Save top titles
    titles_path = f"{AGGREGATIONS_PATH}/top_titles.parquet"
    (
        top_titles.write
        .mode("overwrite")
        .parquet(titles_path)
    )
    print(f"\n  Top titles saved to: {titles_path}")

    # ── Engagement Tier Distribution ───────────
    print("\n" + "-"*55)
    print("  Engagement Tier Distribution")
    print("-"*55)
    tier_dist = engagement_tier_distribution(silver_df)
    tier_dist.show(truncate=False)

    # Save tier distribution
    tier_path = f"{AGGREGATIONS_PATH}/engagement_tier_distribution.parquet"
    (
        tier_dist.write
        .mode("overwrite")
        .parquet(tier_path)
    )
    print(f"\n  Tier distribution saved to: {tier_path}")

    print("\n" + "="*55)
    print("  Stage 3 complete. Aggregations ready.")
    print(f"  Output directory: {AGGREGATIONS_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
