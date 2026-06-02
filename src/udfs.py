"""
Reusable UDF functions for the Reddit Spark Pipeline.
Centralized UDF definitions to ensure consistency and reusability.
"""

from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.udf import udf


# ─────────────────────────────────────────────
# Topic Detection UDF
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
# Array Slice UDF
# ─────────────────────────────────────────────

@udf(returnType=ArrayType(StringType()))
def slice_array(arr, n):
    """
    Return first n elements of array, or all if array is shorter.
    Used for getting top N posts from collect_list results.
    """
    if arr is None:
        return []
    return arr[:n]


# ─────────────────────────────────────────────
# Engagement Tier UDF
# ─────────────────────────────────────────────

@udf(returnType=StringType())
def classify_engagement_tier(engagement_score: float) -> str:
    """
    Classify engagement score into tiers.
    Thresholds can be adjusted based on data distribution.
    """
    if engagement_score >= 50_000:
        return "viral"
    elif engagement_score >= 10_000:
        return "high"
    elif engagement_score >= 1_000:
        return "medium"
    else:
        return "low"


# ─────────────────────────────────────────────
# Text Cleaning UDF
# ─────────────────────────────────────────────

@udf(returnType=StringType())
def clean_text(text: str) -> str:
    """
    Clean text by removing newlines and excessive whitespace.
    """
    if text is None:
        return ""
    import re
    return re.sub(r"[\r\n\t]+", " ", text).strip()
