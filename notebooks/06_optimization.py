"""
Stage 6 — Performance Tuning (Caching & Partitioning)
======================================================
Demonstrates Spark optimization techniques including caching, partitioning,
and query plan analysis to improve pipeline performance.

Key Spark concepts demonstrated:
  - cache / persist
  - repartition / coalesce
  - explain(extended=True)
  - StorageLevel
  - Performance benchmarking with time.time()

Usage:
    python 06_optimization.py
    (Run 02_transform.py first so data/silver/ exists)
"""

import time
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
OPTIMIZATION_PATH = "data/optimization"


# ─────────────────────────────────────────────
# 2. Partitioning by Subreddit
# ─────────────────────────────────────────────

def repartition_by_subreddit(df: DataFrame, num_partitions: int = None) -> DataFrame:
    """
    Repartition silver DataFrame by subreddit.
    This improves performance for queries that filter or group by subreddit.
    
    Shows partition count before/after repartitioning.
    """
    before_partitions = df.rdd.getNumPartitions()
    print(f"    Partitions before repartition: {before_partitions}")
    
    if num_partitions is None:
        # Use number of distinct subreddits as a heuristic
        num_partitions = df.select("subreddit").distinct().count()
    
    df_repartitioned = df.repartition(num_partitions, "subreddit")
    
    after_partitions = df_repartitioned.rdd.getNumPartitions()
    print(f"    Partitions after repartition:  {after_partitions}")
    print(f"    Target partitions: {num_partitions}")
    
    return df_repartitioned


# ─────────────────────────────────────────────
# 3. Caching vs Persist
# ─────────────────────────────────────────────

def benchmark_cache_vs_persist(df: DataFrame) -> dict:
    """
    Compare .cache() vs .persist(StorageLevel.MEMORY_AND_DISK).
    Run 3 queries and measure wall-clock time with and without cache.
    
    Returns timing results for comparison.
    """
    results = {}
    
    # Define test queries
    def query1(df):
        return df.groupBy("subreddit").agg(F.avg("engagement_score")).collect()
    
    def query2(df):
        return df.filter(F.col("engagement_score") > 1000).count()
    
    def query3(df):
        return df.groupBy("hour_of_day").agg(F.count("*")).collect()
    
    # ── Without cache ─────────────────────────
    print("\n    [Benchmark 1] Without cache:")
    start = time.time()
    query1(df)
    query2(df)
    query3(df)
    no_cache_time = time.time() - start
    print(f"      Total time: {no_cache_time:.2f}s")
    results["no_cache"] = no_cache_time
    
    # ── With .cache() ─────────────────────────
    print("\n    [Benchmark 2] With .cache():")
    df_cached = df.cache()
    # Trigger cache by running an action
    df_cached.count()
    start = time.time()
    query1(df_cached)
    query2(df_cached)
    query3(df_cached)
    cache_time = time.time() - start
    print(f"      Total time: {cache_time:.2f}s")
    df_cached.unpersist()
    results["cache"] = cache_time
    
    # ── With .persist(MEMORY_AND_DISK) ─────────
    print("\n    [Benchmark 3] With .persist(MEMORY_AND_DISK):")
    df_persisted = df.persist(StorageLevel.MEMORY_AND_DISK)
    # Trigger persist by running an action
    df_persisted.count()
    start = time.time()
    query1(df_persisted)
    query2(df_persisted)
    query3(df_persisted)
    persist_time = time.time() - start
    print(f"      Total time: {persist_time:.2f}s")
    df_persisted.unpersist()
    results["persist_memory_and_disk"] = persist_time
    
    # ── Summary ───────────────────────────────
    print("\n    [Summary] Performance Comparison:")
    print(f"      No cache:              {no_cache_time:.2f}s (baseline)")
    print(f"      .cache():              {cache_time:.2f}s ({(cache_time/no_cache_time)*100:.1f}% of baseline)")
    print(f"      .persist(MEMORY_DISK): {persist_time:.2f}s ({(persist_time/no_cache_time)*100:.1f}% of baseline)")
    
    return results


# ─────────────────────────────────────────────
# 4. Extended Explain Plan Analysis
# ─────────────────────────────────────────────

def analyze_explain_plan(df: DataFrame):
    """
    Read the extended explain plan and annotate what each Exchange (shuffle) step means.
    Include this as a markdown table in your README.
    """
    print("\n" + "="*55)
    print("  Extended Query Plan Analysis")
    print("="*55)
    
    # Get extended explain plan
    print("\n[1] Extended Explain Plan:")
    print("-"*55)
    plan = df.explain(extended=True)
    
    # Run a complex query to show shuffle operations
    complex_query = (
        df
        .groupBy("subreddit", "hour_of_day")
        .agg(
            F.avg("engagement_score").alias("avg_engagement"),
            F.count("*").alias("post_count")
        )
        .orderBy("subreddit", "hour_of_day")
    )
    
    print("\n[2] Complex Query Plan (with shuffles):")
    print("-"*55)
    complex_query.explain(extended=True)
    
    # Annotate Exchange operations
    print("\n" + "="*55)
    print("  Exchange (Shuffle) Step Annotations")
    print("="*55)
    print("""
| Step Type | Description |
|-----------|-------------|
| Exchange (hashpartitioning) | Redistributes data across partitions based on hash of partition key. Required for operations like groupBy, join, and orderBy when data needs to be co-located. |
| Exchange (singlepartition) | Moves all data to a single partition. Used for operations like collect(), take(), or when a single result is needed. |
| Exchange (roundrobinpartitioning) | Distributes data evenly across partitions in round-robin fashion. Used for repartition() without partition keys. |
| SortMergeJoin | Join strategy where both sides are sorted and merged. Efficient for large datasets after shuffling. |
| BroadcastHashJoin | Join strategy where the smaller side is broadcast to all partitions. No shuffle required for the larger side. |
| ShuffleHashJoin | Join strategy where one or both sides are hashed after shuffling. Used when broadcast is not possible. |
| Aggregate | Performs aggregation (sum, count, avg, etc.) within partitions before final aggregation. |
| Sort | Orders data within or across partitions. Required for orderBy operations. |
| Filter | Applies row-level filtering predicates. Pushed down to data source when possible. |
| Project | Selects and computes columns. Can be pushed down to data source for column pruning. |
    """)


# ─────────────────────────────────────────────
# 5. Coalesce vs Repartition
# ─────────────────────────────────────────────

def demonstrate_coalesce(df: DataFrame):
    """
    Demonstrate the difference between repartition and coalesce.
    - repartition: performs a full shuffle (expensive but can increase partitions)
    - coalesce: minimizes data movement (cheaper, only reduces partitions)
    """
    print("\n" + "="*55)
    print("  Repartition vs Coalesce")
    print("="*55)
    
    initial_partitions = df.rdd.getNumPartitions()
    print(f"\n  Initial partitions: {initial_partitions}")
    
    # Repartition to more partitions (full shuffle)
    df_repartitioned = df.repartition(20)
    repartitioned_count = df_repartitioned.rdd.getNumPartitions()
    print(f"  After repartition(20): {repartitioned_count} partitions (full shuffle)")
    
    # Coalesce to fewer partitions (minimal shuffle)
    df_coalesced = df.coalesce(5)
    coalesced_count = df_coalesced.rdd.getNumPartitions()
    print(f"  After coalesce(5): {coalesced_count} partitions (minimal shuffle)")
    
    print("\n  Key difference:")
    print("  - repartition() can increase or decrease partitions, always does full shuffle")
    print("  - coalesce() can only decrease partitions, minimizes data movement")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("reddit-optimization")
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 6 — Performance Tuning")
    print("="*55)

    # ── Read silver layer ─────────────────────
    print(f"\n[1/4] Reading silver Parquet from: {SILVER_PATH}")
    silver_df = spark.read.parquet(SILVER_PATH)
    print(f"  Rows in silver layer: {silver_df.count():,}")

    # ── Partitioning demo ─────────────────────
    print("\n[2/4] Repartitioning by subreddit...")
    print("-"*55)
    repartitioned_df = repartition_by_subreddit(silver_df)
    
    # Save repartitioned data
    print(f"\n  Saving repartitioned data to: {OPTIMIZATION_PATH}")
    (
        repartitioned_df.write
        .mode("overwrite")
        .parquet(f"{OPTIMIZATION_PATH}/repartitioned_by_subreddit.parquet")
    )

    # ── Caching benchmark ─────────────────────
    print("\n[3/4] Benchmarking cache vs persist...")
    print("-"*55)
    cache_results = benchmark_cache_vs_persist(silver_df)

    # ── Explain plan analysis ─────────────────
    print("\n[4/4] Analyzing query plans...")
    analyze_explain_plan(silver_df)
    
    # ── Coalesce demo ─────────────────────────
    demonstrate_coalesce(silver_df)

    # ── Save optimization results ─────────────
    print(f"\n  Saving optimization results to: {OPTIMIZATION_PATH}")
    
    # Create a summary DataFrame with benchmark results
    import pandas as pd
    results_df = pd.DataFrame([cache_results])
    results_df.to_csv(f"{OPTIMIZATION_PATH}/cache_benchmark.csv", index=False)
    print(f"  Cache benchmark results: {OPTIMIZATION_PATH}/cache_benchmark.csv")

    print("\n" + "="*55)
    print("  Stage 6 complete. Performance tuning applied.")
    print(f"  Output directory: {OPTIMIZATION_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
