"""
Stage 1 — Ingest
================
Pulls top posts from Reddit's public JSON API across multiple subreddits,
defines an explicit Spark schema, loads data into a DataFrame, and writes
the raw result to Parquet as the bronze layer.

No auth required — Reddit exposes public endpoints as JSON.

Usage:
    python 01_ingestion.py

Requirements:
    pip install pyspark requests
"""

import time
import requests
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, IntegerType, DoubleType, BooleanType,
)


# ─────────────────────────────────────────────
# 1. Config
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

POSTS_PER_SUB = 100          # Reddit max per request is 100
BRONZE_PATH   = "data/bronze/reddit_posts.parquet"

# Mimic a browser so Reddit doesn't reject the request
HEADERS = {"User-Agent": "spark-learning-project/1.0 (portfolio project)"}


# ─────────────────────────────────────────────
# 2. Explicit schema
#    Always define schemas explicitly in production —
#    inference is slow and makes types non-deterministic.
# ─────────────────────────────────────────────

POST_SCHEMA = StructType([
    StructField("id",             StringType(),  nullable=False),
    StructField("title",          StringType(),  nullable=True),
    StructField("author",         StringType(),  nullable=True),
    StructField("subreddit",      StringType(),  nullable=False),
    StructField("score",          IntegerType(), nullable=True),
    StructField("num_comments",   IntegerType(), nullable=True),
    StructField("upvote_ratio",   DoubleType(),  nullable=True),
    StructField("created_utc",    LongType(),    nullable=True),   # Unix epoch seconds
    StructField("url",            StringType(),  nullable=True),
    StructField("permalink",      StringType(),  nullable=True),
    StructField("is_self",        BooleanType(), nullable=True),   # text post vs link
    StructField("selftext",       StringType(),  nullable=True),
    StructField("link_flair_text",StringType(),  nullable=True),
    StructField("over_18",        BooleanType(), nullable=True),
    StructField("is_video",       BooleanType(), nullable=True),
    StructField("domain",         StringType(),  nullable=True),
])


# ─────────────────────────────────────────────
# 3. Fetch from Reddit API
# ─────────────────────────────────────────────

def fetch_subreddit(subreddit: str, limit: int = 100) -> list[dict]:
    """
    Fetch top posts from one subreddit using Reddit's public JSON endpoint.
    Returns a flat list of post dictionaries matching POST_SCHEMA.
    """
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"limit": limit, "t": "day"}   # t= : hour | day | week | month | year | all

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
            "id":              d.get("id"),
            "title":           d.get("title"),
            "author":          d.get("author"),
            "subreddit":       subreddit,            # use our normalised name
            "score":           d.get("score"),
            "num_comments":    d.get("num_comments"),
            "upvote_ratio":    d.get("upvote_ratio"),
            "created_utc":     int(d.get("created_utc", 0)),
            "url":             d.get("url"),
            "permalink":       d.get("permalink"),
            "is_self":         d.get("is_self"),
            "selftext":        d.get("selftext"),
            "link_flair_text": d.get("link_flair_text"),
            "over_18":         d.get("over_18"),
            "is_video":        d.get("is_video"),
            "domain":          d.get("domain"),
        })

    return records


def fetch_all_subreddits(subreddits: list[str], limit: int = 100) -> list[dict]:
    """
    Loops over all subreddits with a short sleep to be polite to the API.
    Returns combined list of all post records.
    """
    all_records = []
    for sub in subreddits:
        print(f"  Fetching r/{sub} ...")
        records = fetch_subreddit(sub, limit)
        all_records.extend(records)
        print(f"    → {len(records)} posts fetched")
        time.sleep(1)    # 1s between requests — keeps Reddit happy

    return all_records


# ─────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────

def main():
    # ── 4a. SparkSession ──────────────────────
    spark = (
        SparkSession.builder
        .appName("reddit-ingestion")
        # In local mode, use all available cores
        .master("local[*]")
        # Avoid writing _SUCCESS files (cleaner bronze output)
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .getOrCreate()
    )
    # Suppress INFO logs so output is readable while learning
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 1 — Reddit Ingestion")
    print("="*55)

    # ── 4b. Fetch raw data via API ─────────────
    print("\n[1/4] Fetching posts from Reddit API...")
    raw_records = fetch_all_subreddits(SUBREDDITS, limit=POSTS_PER_SUB)
    print(f"\n  Total records fetched: {len(raw_records)}")

    if not raw_records:
        print("  No records fetched — check your internet connection and try again.")
        spark.stop()
        return

    # ── 4c. Create DataFrame with explicit schema ──
    print("\n[2/4] Creating Spark DataFrame...")
    df = spark.createDataFrame(raw_records, schema=POST_SCHEMA)

    # ── 4d. Exploratory checks ─────────────────
    print("\n[3/4] Schema & preview:")
    df.printSchema()

    print("\n  Sample rows:")
    df.select("subreddit", "title", "score", "num_comments", "upvote_ratio").show(
        n=10, truncate=50, vertical=False
    )

    # Null counts — important to document for downstream stages
    print("\n  Null counts per column:")
    from pyspark.sql.functions import col, sum as spark_sum, isnan, when

    null_counts = df.select([
        spark_sum(
            when(col(c).isNull() | (col(c).cast("string") == ""), 1).otherwise(0)
        ).alias(c)
        for c in df.columns
    ])
    null_counts.show(vertical=True)

    # Row counts per subreddit
    print("\n  Posts per subreddit:")
    df.groupBy("subreddit").count().orderBy("count", ascending=False).show()

    # Score distribution summary
    print("\n  Score distribution:")
    df.select("score", "num_comments", "upvote_ratio").summary(
        "count", "min", "25%", "50%", "75%", "max"
    ).show()

    # ── 4e. Write bronze Parquet ───────────────
    print(f"\n[4/4] Writing bronze Parquet to: {BRONZE_PATH}")
    (
        df.write
        .mode("overwrite")
        .parquet(BRONZE_PATH)
    )
    print("  Done.")

    # Verify write by reading back and counting
    verify_df = spark.read.parquet(BRONZE_PATH)
    print(f"\n  Verification — rows written: {verify_df.count()}")
    print(f"  Partitions in memory: {verify_df.rdd.getNumPartitions()}")

    print("\n" + "="*55)
    print("  Stage 1 complete. Bronze layer ready.")
    print(f"  Output: {BRONZE_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()