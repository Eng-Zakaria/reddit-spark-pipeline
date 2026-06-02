"""
Schema definitions for the Reddit Spark Pipeline.
Centralized schema definitions to ensure consistency across all stages.
"""

from pyspark.sql.types import (
    StructType, StructField,
    StringType, LongType, IntegerType, DoubleType, BooleanType,
)


# ─────────────────────────────────────────────
# Bronze Layer Schema (Raw Reddit Posts)
# ─────────────────────────────────────────────

POST_SCHEMA = StructType([
    StructField("id", StringType(), nullable=False),
    StructField("title", StringType(), nullable=True),
    StructField("author", StringType(), nullable=True),
    StructField("subreddit", StringType(), nullable=False),
    StructField("score", IntegerType(), nullable=True),
    StructField("num_comments", IntegerType(), nullable=True),
    StructField("upvote_ratio", DoubleType(), nullable=True),
    StructField("created_utc", LongType(), nullable=True),
    StructField("url", StringType(), nullable=True),
    StructField("permalink", StringType(), nullable=True),
    StructField("is_self", BooleanType(), nullable=True),
    StructField("selftext", StringType(), nullable=True),
    StructField("link_flair_text", StringType(), nullable=True),
    StructField("over_18", BooleanType(), nullable=True),
    StructField("is_video", BooleanType(), nullable=True),
    StructField("domain", StringType(), nullable=True),
])


# ─────────────────────────────────────────────
# Silver Layer Schema (Cleaned & Enriched)
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
# Category Lookup Schema
# ─────────────────────────────────────────────

CATEGORY_LOOKUP_SCHEMA = StructType([
    StructField("subreddit", StringType(), nullable=False),
    StructField("category", StringType(), nullable=False),
])
