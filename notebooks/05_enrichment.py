"""
Stage 5 — UDFs & Joins (Topic Tagging + Broadcast)
==================================================
Demonstrates custom UDFs, broadcast joins for small lookup tables,
and SQL API usage for topic tagging and category enrichment.

Key Spark concepts demonstrated:
  - UDF / pandas_udf
  - broadcast join
  - join types (inner, left, left_outer)
  - SQL API / createTempView
  - df.explain() for query plan analysis

Usage:
    python 05_enrichment.py
    (Run 02_transform.py first so data/silver/ exists)
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.udf import udf


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
ENRICHMENT_PATH = "data/enrichment"


# ─────────────────────────────────────────────
# 2. Topic Tagging UDF
# ─────────────────────────────────────────────

@udf(returnType=StringType())
def detect_topic(title: str) -> str:
    """
    Python UDF that scans post titles for topic keywords.
    Returns a topic tag based on keyword matching.
    
    Topics: AI, crypto, climate, war, sports, politics, tech, science
    """
    if title is None:
        return "other"
    
    title_lower = title.lower()
    
    # Define keyword patterns for each topic
    topic_keywords = {
        "ai": ["ai", "artificial intelligence", "machine learning", "neural", "gpt", "llm", "chatgpt"],
        "crypto": ["bitcoin", "crypto", "blockchain", "ethereum", "nft", "defi"],
        "climate": ["climate", "global warming", "environment", "carbon", "renewable", "solar"],
        "war": ["war", "conflict", "military", "ukraine", "russia", "israel", "palestine", "gaza"],
        "sports": ["sport", "football", "basketball", "soccer", "nba", "nfl", "olympics", "world cup"],
        "politics": ["politic", "election", "government", "president", "congress", "senate", "vote"],
        "tech": ["tech", "software", "programming", "code", "developer", "app", "startup"],
        "science": ["science", "research", "study", "discovery", "space", "nasa", "physics"],
    }
    
    # Check for each topic's keywords
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in title_lower:
                return topic
    
    return "other"


# ─────────────────────────────────────────────
# 3. Subreddit Category Lookup DataFrame
# ─────────────────────────────────────────────

def create_subreddit_category_lookup(spark: SparkSession) -> DataFrame:
    """
    Create a small lookup DataFrame mapping subreddit → category.
    This will be broadcast-joined onto the main DataFrame.
    """
    data = [
        ("technology", "tech"),
        ("datascience", "tech"),
        ("programming", "tech"),
        ("machinelearning", "tech"),
        ("worldnews", "news"),
        ("business", "news"),
        ("gaming", "gaming"),
        ("science", "science"),
        ("movies", "entertainment"),
        ("todayilearned", "general"),
    ]
    
    schema = ["subreddit", "category"]
    
    return spark.createDataFrame(data, schema)


# ─────────────────────────────────────────────
# 4. Apply Topic Tagging
# ─────────────────────────────────────────────

def apply_topic_tagging(df: DataFrame) -> DataFrame:
    """
    Apply the topic tagging UDF to all posts.
    Registers the UDF and applies it to the title column.
    """
    return df.withColumn(
        "topic",
        detect_topic(F.col("title"))
    )


# ─────────────────────────────────────────────
# 5. Broadcast Join with Category Lookup
# ─────────────────────────────────────────────

def broadcast_join_category(df: DataFrame, lookup_df: DataFrame) -> DataFrame:
    """
    Broadcast-join the category lookup onto the main DataFrame.
    Broadcast joins are efficient when one side is small enough to fit in memory.
    """
    from pyspark.sql.functions import broadcast
    
    return (
        df
        .join(
            broadcast(lookup_df),
            on="subreddit",
            how="left"
        )
    )


# ─────────────────────────────────────────────
# 6. Compare Join Strategies
# ─────────────────────────────────────────────

def compare_join_strategies(df: DataFrame, lookup_df: DataFrame):
    """
    Compare df.explain() with and without broadcast.
    This demonstrates the performance difference between broadcast and shuffle joins.
    """
    print("\n" + "="*55)
    print("  Query Plan Comparison: Broadcast vs Shuffle Join")
    print("="*55)
    
    # Broadcast join plan
    print("\n[1] Broadcast Join Query Plan:")
    print("-"*55)
    broadcast_join = df.join(
        F.broadcast(lookup_df),
        on="subreddit",
        how="left"
    )
    broadcast_join.explain(extended=False)
    
    # Regular shuffle join plan
    print("\n[2] Regular Shuffle Join Query Plan:")
    print("-"*55)
    shuffle_join = df.join(
        lookup_df,
        on="subreddit",
        how="left"
    )
    shuffle_join.explain(extended=False)
    
    print("\n" + "="*55)
    print("  Note: Broadcast join shows 'BroadcastHashJoin' operator")
    print("  Regular join shows 'SortMergeJoin' or 'ShuffleHashJoin'")
    print("="*55)


# ─────────────────────────────────────────────
# 7. SQL API Demo
# ─────────────────────────────────────────────

def demo_sql_api(df: DataFrame):
    """
    Register DataFrame as a temp view and run SQL queries.
    Demonstrates SQL fluency alongside DataFrame API.
    """
    print("\n" + "="*55)
    print("  SQL API Demo")
    print("="*55)
    
    # Register as temp view
    df.createOrReplaceTempView("reddit_posts")
    
    # Run SQL query for trending posts by topic
    print("\n[SQL] Top topics by average engagement:")
    print("-"*55)
    topic_query = """
        SELECT 
            topic,
            COUNT(*) as post_count,
            AVG(engagement_score) as avg_engagement,
            MAX(engagement_score) as max_engagement
        FROM reddit_posts
        WHERE topic != 'other'
        GROUP BY topic
        ORDER BY avg_engagement DESC
        LIMIT 10
    """
    spark = df.sparkSession
    spark.sql(topic_query).show(truncate=False)
    
    # Run SQL query for category analysis
    print("\n[SQL] Posts per category:")
    print("-"*55)
    category_query = """
        SELECT 
            category,
            COUNT(*) as post_count,
            AVG(engagement_score) as avg_engagement,
            AVG(score) as avg_score
        FROM reddit_posts
        GROUP BY category
        ORDER BY post_count DESC
    """
    spark.sql(category_query).show(truncate=False)
    
    # Run SQL query combining topic and category
    print("\n[SQL] Topic distribution within each category:")
    print("-"*55)
    combined_query = """
        SELECT 
            category,
            topic,
            COUNT(*) as post_count,
            AVG(engagement_score) as avg_engagement
        FROM reddit_posts
        WHERE topic != 'other'
        GROUP BY category, topic
        ORDER BY category, post_count DESC
    """
    spark.sql(combined_query).show(30, truncate=False)


# ─────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("reddit-enrichment")
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10MB
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 5 — UDFs & Joins (Topic Tagging + Broadcast)")
    print("="*55)

    # ── Read silver layer ─────────────────────
    print(f"\n[1/5] Reading silver Parquet from: {SILVER_PATH}")
    silver_df = spark.read.parquet(SILVER_PATH)
    print(f"  Rows in silver layer: {silver_df.count():,}")

    # ── Create category lookup ─────────────────
    print("\n[2/5] Creating subreddit category lookup...")
    lookup_df = create_subreddit_category_lookup(spark)
    print("  Category lookup DataFrame:")
    lookup_df.show(truncate=False)

    # ── Compare join strategies ────────────────
    print("\n[3/5] Comparing broadcast vs shuffle join query plans...")
    compare_join_strategies(silver_df, lookup_df)

    # ── Apply topic tagging UDF ─────────────────
    print("\n[4/5] Applying topic tagging UDF...")
    df_with_topic = apply_topic_tagging(silver_df)
    
    print("\n  Topic distribution:")
    (
        df_with_topic
        .groupBy("topic")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("engagement_score").alias("avg_engagement")
        )
        .orderBy(F.desc("post_count"))
        .show(truncate=False)
    )

    # ── Broadcast join category lookup ─────────
    print("\n[5/5] Applying broadcast join for category enrichment...")
    enriched_df = broadcast_join_category(df_with_topic, lookup_df)
    
    print("\n  Sample enriched rows:")
    (
        enriched_df
        .select(
            "subreddit", "category", "topic", "title",
            "engagement_score", "engagement_tier"
        )
        .show(10, truncate=40)
    )

    # ── SQL API demo ───────────────────────────
    demo_sql_api(enriched_df)

    # ── Save enriched data ─────────────────────
    print(f"\n  Saving enriched data to: {ENRICHMENT_PATH}")
    (
        enriched_df.write
        .mode("overwrite")
        .parquet(f"{ENRICHMENT_PATH}/enriched_posts.parquet")
    )
    
    # Save topic summary
    topic_summary = (
        enriched_df
        .groupBy("topic")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("engagement_score").alias("avg_engagement"),
            F.max("engagement_score").alias("max_engagement"),
        )
        .orderBy(F.desc("post_count"))
    )
    (
        topic_summary.write
        .mode("overwrite")
        .parquet(f"{ENRICHMENT_PATH}/topic_summary.parquet")
    )
    
    # Save category summary
    category_summary = (
        enriched_df
        .groupBy("category")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("engagement_score").alias("avg_engagement"),
            F.avg("score").alias("avg_score"),
        )
        .orderBy(F.desc("post_count"))
    )
    (
        category_summary.write
        .mode("overwrite")
        .parquet(f"{ENRICHMENT_PATH}/category_summary.parquet")
    )
    
    print(f"  Enriched posts: {ENRICHMENT_PATH}/enriched_posts.parquet")
    print(f"  Topic summary: {ENRICHMENT_PATH}/topic_summary.parquet")
    print(f"  Category summary: {ENRICHMENT_PATH}/category_summary.parquet")

    print("\n" + "="*55)
    print("  Stage 5 complete. UDFs and joins applied.")
    print(f"  Output directory: {ENRICHMENT_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
