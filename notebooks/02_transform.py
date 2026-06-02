"""
Stage 2 — Clean & Enrich (Silver Layer)
========================================
Reads the bronze Parquet, applies data quality rules, derives new columns,
and writes a clean, enriched DataFrame as the silver layer.

Key Spark concepts demonstrated:
  - filter / where
  - withColumn + cast
  - to_timestamp / from_unixtime
  - date_format / hour / dayofweek
  - when / otherwise  (multi-branch logic)
  - isNull / fillna / dropDuplicates
  - regexp_replace / lower / trim  (string functions)
  - lit                            (literal column values)
  - col                            (column references)

Usage:
    python 02_transform.py
    (Run 01_ingestion.py first so data/bronze/ exists)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

BRONZE_PATH = "data/bronze/reddit_posts.parquet"
SILVER_PATH = "data/silver/reddit_posts_clean.parquet"

# Engagement score weights — document your decisions
COMMENT_WEIGHT   = 3    # comments signal deeper engagement than upvotes
MIN_SCORE        = 1    # drop posts with zero or negative scores (removed/spam)
MIN_TITLE_LENGTH = 5    # drop nonsense one-word titles
BRIGADING_RATIO  = 0.5  # upvote_ratio below this → flag as suspicious


# ─────────────────────────────────────────────
# 2. Quality filters
#    Each filter is a separate step so you can
#    comment one out and re-run without rewriting.
# ─────────────────────────────────────────────

def drop_bad_rows(df: DataFrame) -> DataFrame:
    """
    Remove rows that are structurally unusable downstream.
    Log counts so you can see what was dropped.
    """
    before = df.count()

    df = (
        df
        # Must have a title — it's our primary text field
        .filter(F.col("title").isNotNull())
        .filter(F.length(F.trim(F.col("title"))) >= MIN_TITLE_LENGTH)

        # Must have a valid score
        .filter(F.col("score").isNotNull())
        .filter(F.col("score") >= MIN_SCORE)

        # Must have a real author (deleted/removed posts)
        .filter(F.col("author").isNotNull())
        .filter(~F.col("author").isin("[deleted]", "[removed]", "AutoModerator"))

        # Must have a valid timestamp
        .filter(F.col("created_utc").isNotNull())
        .filter(F.col("created_utc") > 0)

        # Deduplicate — same post can appear in multiple API calls
        .dropDuplicates(["id"])
    )

    after = df.count()
    print(f"    Rows before quality filter : {before:,}")
    print(f"    Rows after  quality filter : {after:,}")
    print(f"    Rows dropped               : {before - after:,}")
    return df


# ─────────────────────────────────────────────
# 3. Timestamp & time-based derived columns
# ─────────────────────────────────────────────

def add_time_columns(df: DataFrame) -> DataFrame:
    """
    Convert Unix epoch to timestamp and derive time features.
    These drive the hourly/daily aggregations in Stage 3.

    created_utc is seconds since epoch (long) → cast to timestamp.
    """
    return (
        df
        # Cast epoch seconds to proper TimestampType
        .withColumn(
            "post_timestamp",
            F.to_timestamp(F.from_unixtime(F.col("created_utc")))
        )

        # Hour of day: 0–23 — used to find peak posting hours
        .withColumn("hour_of_day", F.hour("post_timestamp"))

        # Day of week: 1 = Sunday, 7 = Saturday (Spark default)
        .withColumn("day_of_week_num", F.dayofweek("post_timestamp"))

        # Human-readable day name for charts and README output
        .withColumn(
            "day_of_week",
            F.date_format("post_timestamp", "EEEE")   # "Monday", "Tuesday" …
        )

        # Date only — useful for daily groupBy aggregations
        .withColumn("post_date", F.to_date("post_timestamp"))

        # How old is this post in hours relative to when the pipeline ran?
        # F.current_timestamp() returns the pipeline execution time
        .withColumn(
            "post_age_hours",
            (
                F.unix_timestamp(F.current_timestamp()) -
                F.col("created_utc")
            ).cast("double") / 3600
        )
    )


# ─────────────────────────────────────────────
# 4. Engagement score & tiering
# ─────────────────────────────────────────────

def add_engagement_columns(df: DataFrame) -> DataFrame:
    """
    Derive a composite engagement score and bucket posts into tiers.

    engagement_score = score + (num_comments × COMMENT_WEIGHT)
    This rewards posts that spark conversation, not just passive upvotes.

    Tiers are absolute thresholds — adjust after looking at your data's
    score distribution from Stage 1.
    """
    df = df.withColumn(
        "num_comments_clean",
        F.coalesce(F.col("num_comments"), F.lit(0)).cast(IntegerType())
    )

    df = df.withColumn(
        "engagement_score",
        F.col("score") + (F.col("num_comments_clean") * F.lit(COMMENT_WEIGHT))
    )

    # comment_ratio: how discussion-heavy is the post vs pure upvotes?
    df = df.withColumn(
        "comment_ratio",
        F.when(F.col("score") > 0,
               F.col("num_comments_clean").cast("double") / F.col("score")
        ).otherwise(F.lit(0.0))
    )

    # Engagement tier using when/otherwise — a key Spark pattern
    df = df.withColumn(
        "engagement_tier",
        F.when(F.col("engagement_score") >= 50_000, F.lit("viral"))
         .when(F.col("engagement_score") >= 10_000, F.lit("high"))
         .when(F.col("engagement_score") >= 1_000,  F.lit("medium"))
         .otherwise(F.lit("low"))
    )

    return df


# ─────────────────────────────────────────────
# 5. Quality & anomaly flags
# ─────────────────────────────────────────────

def add_flag_columns(df: DataFrame) -> DataFrame:
    """
    Boolean flags that downstream consumers can filter on.
    Keeping these as flags (not dropping rows) preserves data for analysis —
    you might WANT to study brigaded posts, not just exclude them.
    """
    return (
        df
        # Suspicious vote manipulation — low ratio despite high score
        .withColumn(
            "is_brigaded",
            F.when(
                (F.col("upvote_ratio") < BRIGADING_RATIO) &
                (F.col("score") > 100),
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # Posts with no body text (link posts vs self posts)
        .withColumn(
            "is_link_post",
            F.when(
                F.col("is_self") == False,
                F.lit(True)
            ).otherwise(F.lit(False))
        )

        # Unusually high comment-to-score ratio → controversial post
        .withColumn(
            "is_controversial",
            F.when(F.col("comment_ratio") > 1.0, F.lit(True))
             .otherwise(F.lit(False))
        )

        # External domain (not reddit self-post or i.redd.it)
        .withColumn(
            "is_external_link",
            F.when(
                ~F.col("domain").isin(
                    "self", "i.redd.it", "v.redd.it", "reddit.com",
                    "i.imgur.com", "imgur.com"
                ) & F.col("domain").isNotNull(),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
    )


# ─────────────────────────────────────────────
# 6. Text cleaning
# ─────────────────────────────────────────────

def clean_text_columns(df: DataFrame) -> DataFrame:
    """
    Normalise string columns for consistent downstream matching.
    Lower-casing and trimming prevents group-by mismatches like
    "Python" vs "python" being treated as different values.
    """
    return (
        df
        # Normalise subreddit casing (API can return mixed case)
        .withColumn("subreddit", F.lower(F.trim(F.col("subreddit"))))

        # Strip whitespace from author
        .withColumn("author", F.trim(F.col("author")))

        # Clean selftext: remove newlines and excessive whitespace
        .withColumn(
            "selftext_clean",
            F.regexp_replace(
                F.coalesce(F.col("selftext"), F.lit("")),
                r"[\r\n\t]+", " "
            )
        )

        # Title word count — useful feature for ML stage later
        .withColumn(
            "title_word_count",
            F.size(F.split(F.trim(F.col("title")), r"\s+"))
        )

        # Fill nulls in non-critical string columns
        .fillna({
            "link_flair_text": "none",
            "domain":          "unknown",
        })
    )


# ─────────────────────────────────────────────
# 7. Select final silver columns
#    Be explicit — don't carry every raw column forward.
#    This is good pipeline hygiene and keeps Parquet files lean.
# ─────────────────────────────────────────────

SILVER_COLUMNS = [
    # Identity
    "id", "subreddit", "author",

    # Content
    "title", "title_word_count", "selftext_clean",
    "url", "permalink", "domain",
    "link_flair_text",

    # Time
    "created_utc", "post_timestamp", "post_date",
    "hour_of_day", "day_of_week", "day_of_week_num", "post_age_hours",

    # Engagement
    "score", "num_comments_clean", "upvote_ratio",
    "engagement_score", "comment_ratio", "engagement_tier",

    # Flags
    "is_self", "is_video", "over_18",
    "is_link_post", "is_brigaded", "is_controversial", "is_external_link",
]


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("reddit-transform")
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 2 — Clean & Enrich")
    print("="*55)

    # ── 8a. Read bronze ───────────────────────
    print(f"\n[1/6] Reading bronze Parquet from: {BRONZE_PATH}")
    bronze_df = spark.read.parquet(BRONZE_PATH)
    print(f"  Rows in bronze layer: {bronze_df.count():,}")

    # ── 8b. Drop bad rows ─────────────────────
    print("\n[2/6] Applying quality filters...")
    df = drop_bad_rows(bronze_df)

    # ── 8c. Add time columns ──────────────────
    print("\n[3/6] Adding time-derived columns...")
    df = add_time_columns(df)

    # ── 8d. Engagement score & tiers ──────────
    print("\n[4/6] Computing engagement score and tiers...")
    df = add_engagement_columns(df)

    # ── 8e. Flags & text cleaning ─────────────
    print("\n[5/6] Adding quality flags and cleaning text...")
    df = add_flag_columns(df)
    df = clean_text_columns(df)

    # Final column selection
    df = df.select(SILVER_COLUMNS)

    # ── 8f. Spot checks before writing ────────
    print("\n  Engagement tier distribution:")
    df.groupBy("engagement_tier").count() \
      .orderBy("count", ascending=False).show()

    print("\n  Avg engagement score per subreddit:")
    df.groupBy("subreddit") \
      .agg(
          F.avg("engagement_score").alias("avg_engagement"),
          F.max("engagement_score").alias("max_engagement"),
          F.count("*").alias("post_count"),
      ) \
      .orderBy("avg_engagement", ascending=False) \
      .show()

    print("\n  Flag summary (count of True per flag):")
    flag_cols = ["is_brigaded", "is_controversial", "is_external_link", "is_video", "over_18"]
    df.select([F.sum(F.col(c).cast("int")).alias(c) for c in flag_cols]).show()

    print("\n  Sample silver rows:")
    df.select(
        "subreddit", "title", "engagement_score",
        "engagement_tier", "hour_of_day", "day_of_week", "is_brigaded"
    ).show(10, truncate=45)

    print("\n  Silver schema:")
    df.printSchema()

    # ── 8g. Write silver Parquet ──────────────
    print(f"\n[6/6] Writing silver Parquet to: {SILVER_PATH}")
    (
        df.write
        .mode("overwrite")
        .parquet(SILVER_PATH)
    )

    # Verify
    verify_df = spark.read.parquet(SILVER_PATH)
    print(f"\n  Verification — rows written  : {verify_df.count():,}")
    print(f"  Columns in silver layer      : {len(verify_df.columns)}")

    print("\n" + "="*55)
    print("  Stage 2 complete. Silver layer ready.")
    print(f"  Output: {SILVER_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()