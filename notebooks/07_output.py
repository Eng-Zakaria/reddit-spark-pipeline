"""
Stage 7 — Output (Gold Layer & Visualization)
==============================================
Writes the final gold layer as partitioned Parquet files and generates
visualizations for the README and portfolio.

Key Spark concepts demonstrated:
  - write.parquet / partitionBy
  - read with predicate pushdown
  - toPandas() for visualization
  - matplotlib / seaborn for charts

Usage:
    python 07_output.py
    (Run 02_transform.py first so data/silver/ exists)
"""

import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

SILVER_PATH = "data/silver/reddit_posts_clean.parquet"
GOLD_PATH = "data/gold"
CHARTS_PATH = "charts"


# ─────────────────────────────────────────────
# 2. Write Gold Layer with Partitioning
# ─────────────────────────────────────────────

def write_gold_layer(df: DataFrame):
    """
    Write gold layer as Parquet partitioned by subreddit and day_of_week.
    Partitioning improves query performance for filters on these columns.
    """
    print("\n[1/3] Writing gold layer with partitioning...")
    print(f"  Output path: {GOLD_PATH}")
    print(f"  Partitioned by: subreddit, day_of_week")
    
    (
        df
        .write
        .mode("overwrite")
        .partitionBy("subreddit", "day_of_week")
        .parquet(GOLD_PATH)
    )
    
    print("  Gold layer written successfully.")


# ─────────────────────────────────────────────
# 3. Read with Predicate Pushdown
# ─────────────────────────────────────────────

def demonstrate_predicate_pushdown(spark: SparkSession):
    """
    Read back with a filter and verify predicate pushdown in explain().
    Predicate pushdown allows Spark to skip reading irrelevant partitions.
    """
    print("\n[2/3] Demonstrating predicate pushdown...")
    
    # Read with filter on partition column
    print("\n  Reading with filter: subreddit = 'datascience'")
    filtered_df = spark.read.parquet(GOLD_PATH).filter("subreddit = 'datascience'")
    
    print("\n  Query plan (note PushedFilters):")
    filtered_df.explain(extended=False)
    
    count = filtered_df.count()
    print(f"\n  Rows returned: {count:,}")
    
    return filtered_df


# ─────────────────────────────────────────────
# 4. Generate Aggregations for Visualization
# ─────────────────────────────────────────────

def prepare_visualization_data(df: DataFrame):
    """
    Prepare aggregated data for visualization.
    Returns DataFrames for different chart types.
    """
    # Subreddit engagement bar chart data
    subreddit_engagement = (
        df
        .groupBy("subreddit")
        .agg(
            F.avg("engagement_score").alias("avg_engagement"),
            F.count("*").alias("post_count"),
        )
        .orderBy(F.desc("avg_engagement"))
    )
    
    # Hourly post heatmap data
    hourly_heatmap = (
        df
        .groupBy("subreddit", "hour_of_day")
        .agg(F.count("*").alias("post_count"))
        .orderBy("subreddit", "hour_of_day")
    )
    
    # Trending posts table data
    trending_posts = (
        df
        .select(
            "subreddit", "title", "engagement_score", "score",
            "num_comments_clean", "engagement_tier", "hour_of_day"
        )
        .orderBy(F.desc("engagement_score"))
        .limit(20)
    )
    
    return subreddit_engagement, hourly_heatmap, trending_posts


# ─────────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────────

def create_visualizations(subreddit_engagement_df, hourly_heatmap_df, trending_posts_df):
    """
    Generate 3 charts using matplotlib and seaborn:
    1. Subreddit engagement bar chart
    2. Hourly post heatmap
    3. Trending posts table (as styled DataFrame)
    
    Export all as PNGs for README and LinkedIn.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Create charts directory if it doesn't exist
    os.makedirs(CHARTS_PATH, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # ── Chart 1: Subreddit Engagement Bar Chart ──
    print("\n  [Chart 1] Creating subreddit engagement bar chart...")
    sub_pd = subreddit_engagement_df.toPandas()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(sub_pd['subreddit'], sub_pd['avg_engagement'], color='steelblue')
    ax.set_xlabel('Average Engagement Score', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.set_title('Average Engagement Score by Subreddit', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    chart1_path = os.path.join(CHARTS_PATH, "subreddit_engagement_bar_chart.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {chart1_path}")
    
    # ── Chart 2: Hourly Post Heatmap ─────────────
    print("\n  [Chart 2] Creating hourly post heatmap...")
    hourly_pd = hourly_heatmap_df.toPandas()
    
    # Pivot for heatmap
    heatmap_data = hourly_pd.pivot(index='subreddit', columns='hour_of_day', values='post_count')
    heatmap_data = heatmap_data.fillna(0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                cbar_kws={'label': 'Post Count'}, ax=ax)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.set_title('Hourly Post Volume Heatmap by Subreddit', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    chart2_path = os.path.join(CHARTS_PATH, "hourly_post_heatmap.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {chart2_path}")
    
    # ── Chart 3: Trending Posts Table ────────────
    print("\n  [Chart 3] Creating trending posts table...")
    trending_pd = trending_posts_df.toPandas()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = trending_pd[['subreddit', 'title', 'engagement_score', 
                              'score', 'num_comments_clean', 'engagement_tier']]
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                     cellLoc='left', loc='center', colWidths=[0.1, 0.4, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Top 20 Trending Posts by Engagement Score', 
                 fontsize=14, fontweight='bold', pad=20)
    
    chart3_path = os.path.join(CHARTS_PATH, "trending_posts_table.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {chart3_path}")
    
    # Also save as CSV for reference
    csv_path = os.path.join(CHARTS_PATH, "trending_posts.csv")
    trending_pd.to_csv(csv_path, index=False)
    print(f"    CSV saved: {csv_path}")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("reddit-output")
        .master("local[*]")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "="*55)
    print("  Stage 7 — Output (Gold Layer & Visualization)")
    print("="*55)

    # ── Read silver layer ─────────────────────
    print(f"\n[1/3] Reading silver Parquet from: {SILVER_PATH}")
    silver_df = spark.read.parquet(SILVER_PATH)
    print(f"  Rows in silver layer: {silver_df.count():,}")

    # ── Write gold layer ───────────────────────
    write_gold_layer(silver_df)

    # ── Demonstrate predicate pushdown ─────────
    filtered_df = demonstrate_predicate_pushdown(spark)

    # ── Prepare visualization data ─────────────
    print("\n[3/3] Preparing data for visualization...")
    subreddit_engagement, hourly_heatmap, trending_posts = prepare_visualization_data(silver_df)
    
    print("\n  Sample subreddit engagement data:")
    subreddit_engagement.show(truncate=False)
    
    print("\n  Sample hourly heatmap data:")
    hourly_heatmap.show(10, truncate=False)
    
    print("\n  Sample trending posts:")
    trending_posts.show(10, truncate=50)

    # ── Create visualizations ─────────────────
    print("\n" + "-"*55)
    print("  Generating visualizations...")
    print("-"*55)
    create_visualizations(subreddit_engagement, hourly_heatmap, trending_posts)

    print("\n" + "="*55)
    print("  Stage 7 complete.")
    print(f"  Gold layer: {GOLD_PATH}")
    print(f"  Charts: {CHARTS_PATH}")
    print("="*55 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
