"""Data loading functions with Streamlit caching."""

import streamlit as st
import pandas as pd
from pathlib import Path

from src.db import get_db


@st.cache_data(ttl=300)
def load_overview_stats(version_id=None):
    """Load overview statistics for a specific version."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Total articles
            cur.execute(f"SELECT COUNT(*) as count FROM {schema}.news_articles")
            total_articles = cur.fetchone()["count"]

            # Articles about Ditwah cyclone
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
            """)
            ditwah_articles = cur.fetchone()["count"]

            # Articles by source
            cur.execute(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.news_articles
                GROUP BY source_id
                ORDER BY count DESC
            """)
            by_source = cur.fetchall()

            # Ditwah articles by source
            cur.execute(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
                GROUP BY source_id
                ORDER BY count DESC
            """)
            ditwah_by_source = cur.fetchall()

            if version_id:
                # Total topics for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.topics WHERE topic_id != -1 AND result_version_id = %s",
                    (version_id,)
                )
                total_topics = cur.fetchone()["count"]

                # Total clusters for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE result_version_id = %s",
                    (version_id,)
                )
                total_clusters = cur.fetchone()["count"]

                # Multi-source clusters for this version
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.event_clusters WHERE sources_count > 1 AND result_version_id = %s",
                    (version_id,)
                )
                multi_source = cur.fetchone()["count"]
            else:
                # Fallback for no version selected
                total_topics = 0
                total_clusters = 0
                multi_source = 0

            # Date range
            cur.execute(f"""
                SELECT MIN(date_posted)::date as min_date, MAX(date_posted)::date as max_date
                FROM {schema}.news_articles
            """)
            date_range = cur.fetchone()

    return {
        "total_articles": total_articles,
        "ditwah_articles": ditwah_articles,
        "by_source": by_source,
        "ditwah_by_source": ditwah_by_source,
        "total_topics": total_topics,
        "total_clusters": total_clusters,
        "multi_source_clusters": multi_source,
        "date_range": date_range
    }


@st.cache_data(ttl=300)
def load_topics(version_id=None):
    """Load topic data for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT topic_id, name, description, article_count
                FROM {schema}.topics
                WHERE topic_id != -1 AND result_version_id = %s
                ORDER BY article_count DESC
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_by_source(model_type: str):
    """Load average sentiment by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment,
                    STDDEV(sa.overall_sentiment) as stddev_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
                GROUP BY n.source_id
                ORDER BY avg_sentiment DESC
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_distribution(model_type: str):
    """Load sentiment distribution for box plots."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.source_id,
                    sa.overall_sentiment
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_percentage_by_source(model_type: str):
    """Load sentiment percentage distribution by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.source_id,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment < -0.5) as negative_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment >= -0.5 AND sa.overall_sentiment <= 0.5) as neutral_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment > 0.5) as positive_count,
                    COUNT(*) as total_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
                GROUP BY n.source_id
                ORDER BY n.source_id
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_timeline(model_type: str):
    """Load sentiment over time."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    DATE_TRUNC('day', n.date_posted) as date,
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                WHERE sa.model_type = %s
                GROUP BY DATE_TRUNC('day', n.date_posted), n.source_id
                ORDER BY date
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_sentiment(model_type: str):
    """Load sentiment by topic."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    t.name as topic,
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE sa.model_type = %s AND t.topic_id != -1
                GROUP BY t.name, n.source_id
                HAVING COUNT(*) >= 5
            """, (model_type,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_available_models():
    """Get list of models with analysis results."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT model_type, COUNT(*) as article_count
                FROM {schema}.sentiment_analyses
                GROUP BY model_type
                ORDER BY model_type
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_list():
    """Get list of topics for dropdown."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT name, article_count
                FROM {schema}.topics
                WHERE topic_id != -1
                ORDER BY article_count DESC
                LIMIT 50
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_by_source_topic(model_type: str, topic: str = None):
    """Load sentiment by source, optionally filtered by topic."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            query = f"""
                SELECT
                    n.source_id,
                    AVG(sa.overall_sentiment) as avg_sentiment,
                    STDDEV(sa.overall_sentiment) as stddev_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
            """

            if topic and topic != "All Topics":
                query += f"""
                    JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                    JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                    WHERE sa.model_type = %s AND t.name = %s
                """
                params = (model_type, topic)
            else:
                query += " WHERE sa.model_type = %s"
                params = (model_type,)

            query += " GROUP BY n.source_id ORDER BY avg_sentiment DESC"

            cur.execute(query, params)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_sentiment_percentage_by_source_topic(model_type: str, topic: str = None):
    """Load sentiment percentage distribution by source with optional topic filter."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            query = f"""
                SELECT
                    n.source_id,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment < -0.5) as negative_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment >= -0.5 AND sa.overall_sentiment <= 0.5) as neutral_count,
                    COUNT(*) FILTER (WHERE sa.overall_sentiment > 0.5) as positive_count,
                    COUNT(*) as total_count
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
            """

            if topic and topic != "All Topics":
                query += f"""
                    JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                    JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                    WHERE sa.model_type = %s AND t.name = %s
                """
                params = (model_type, topic)
            else:
                query += " WHERE sa.model_type = %s"
                params = (model_type,)

            query += " GROUP BY n.source_id ORDER BY n.source_id"

            cur.execute(query, params)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_multi_model_comparison(models: list, topic: str = None):
    """Load sentiment data for multiple models with optional topic filter."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            query = f"""
                SELECT
                    sa.model_type,
                    n.source_id,
                    sa.overall_sentiment,
                    t.name as topic
                FROM {schema}.sentiment_analyses sa
                JOIN {schema}.news_articles n ON sa.article_id = n.id
                LEFT JOIN {schema}.article_analysis aa ON sa.article_id = aa.article_id
                LEFT JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE sa.model_type = ANY(%s)
            """
            params = [models]

            if topic and topic != "All Topics":
                query += " AND t.name = %s"
                params.append(topic)

            cur.execute(query, tuple(params))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_by_source(version_id=None):
    """Load topic distribution by source for a specific version."""
    if not version_id:
        return []
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT t.name as topic, n.source_id, COUNT(*) as count
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                JOIN {schema}.news_articles n ON aa.article_id = n.id
                WHERE t.topic_id != -1
                  AND aa.result_version_id = %s
                  AND t.result_version_id = %s
                GROUP BY t.name, n.source_id
                ORDER BY count DESC
            """, (version_id, version_id))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_topic_coverage_by_source(topic_name: str, version_id: str):
    """Load coverage statistics for a specific topic across all sources.

    Args:
        topic_name: The topic name to analyze
        version_id: The topic version ID

    Returns:
        List of dicts with keys: source_id, article_count, source_total, percentage
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                WITH topic_counts AS (
                    SELECT n.source_id, COUNT(*) as topic_count
                    FROM {schema}.article_analysis aa
                    JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                    JOIN {schema}.news_articles n ON aa.article_id = n.id
                    WHERE t.name = %s
                      AND t.topic_id != -1
                      AND aa.result_version_id = %s
                      AND t.result_version_id = %s
                    GROUP BY n.source_id
                ),
                source_totals AS (
                    SELECT n.source_id, COUNT(*) as total_count
                    FROM {schema}.news_articles n
                    WHERE n.is_ditwah_cyclone = 1
                    GROUP BY n.source_id
                )
                SELECT
                    COALESCE(tc.source_id, st.source_id) as source_id,
                    COALESCE(tc.topic_count, 0) as article_count,
                    st.total_count as source_total,
                    ROUND(COALESCE(tc.topic_count, 0) * 100.0 / st.total_count, 1) as percentage
                FROM source_totals st
                LEFT JOIN topic_counts tc ON st.source_id = tc.source_id
                ORDER BY st.source_id
            """, (topic_name, version_id, version_id))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_top_events(version_id=None, limit=20):
    """Load top event clusters for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id, ec.cluster_name, ec.article_count, ec.sources_count,
                       ec.date_start, ec.date_end
                FROM {schema}.event_clusters ec
                WHERE ec.result_version_id = %s
                ORDER BY ec.article_count DESC
                LIMIT {limit}
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_event_details(event_id, version_id=None):
    """Load details for a specific event cluster."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Get articles in cluster
            cur.execute(f"""
                SELECT n.title, n.source_id, n.date_posted, n.url
                FROM {schema}.article_clusters ac
                JOIN {schema}.news_articles n ON ac.article_id = n.id
                WHERE ac.cluster_id = %s
                ORDER BY n.date_posted
            """, (event_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_coverage_timeline():
    """Load daily article counts by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT date_posted::date as date, source_id, COUNT(*) as count
                FROM {schema}.news_articles
                WHERE date_posted IS NOT NULL
                GROUP BY date_posted::date, source_id
                ORDER BY date
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_ditwah_timeline():
    """Load daily Ditwah article counts by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT date_posted::date as date, source_id, COUNT(*) as count
                FROM {schema}.news_articles
                WHERE date_posted IS NOT NULL AND is_ditwah_cyclone = 1
                GROUP BY date_posted::date, source_id
                ORDER BY date
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_article_lengths():
    """Load article lengths for distribution analysis."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    source_id,
                    LENGTH(content) as article_length
                FROM {schema}.news_articles
                WHERE content IS NOT NULL
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_ditwah_article_lengths():
    """Load article lengths for Ditwah articles."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    source_id,
                    LENGTH(content) as article_length
                FROM {schema}.news_articles
                WHERE content IS NOT NULL AND is_ditwah_cyclone = 1
            """)
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_word_frequencies(version_id=None, limit=50):
    """Load word frequencies for a specific version."""
    if not version_id:
        return {}

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT source_id, word, frequency, tfidf_score, rank
                FROM {schema}.word_frequencies
                WHERE result_version_id = %s
                  AND rank <= %s
                ORDER BY source_id, rank
            """, (version_id, limit))
            rows = cur.fetchall()

            # Group by source
            result = {}
            for row in rows:
                source = row['source_id']
                if source not in result:
                    result[source] = []
                result[source].append(row)
            return result


@st.cache_resource
def load_bertopic_model(version_id=None):
    """Load the saved BERTopic model for a specific version.

    Tries to load from database first (for team collaboration),
    then falls back to filesystem for backward compatibility.
    """
    if not version_id:
        return None

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    # Get the embedding model from version configuration
    from src.versions import get_version_config
    config = get_version_config(version_id)
    embedding_model_name = "all-mpnet-base-v2"  # default
    if config and "embeddings" in config and "model" in config["embeddings"]:
        embedding_model_name = config["embeddings"]["model"]

    # Load the embedding model
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        st.warning(f"Failed to load embedding model '{embedding_model_name}': {e}")
        return None

    # Strategy 1: Try loading from database
    from src.versions import get_model_from_version
    import tempfile

    try:
        # Extract model from database to temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"bertopic_{version_id[:8]}_")
        model_path = get_model_from_version(version_id, temp_dir)

        if model_path:
            try:
                model = BERTopic.load(model_path, embedding_model=embedding_model)
                return model
            except Exception as e:
                st.warning(f"Model found in database but failed to load: {e}")
    except Exception:
        # Database loading failed, will try filesystem
        pass

    # Strategy 2: Fallback to filesystem (backward compatibility)
    model_path = Path(__file__).parent.parent.parent / "models" / f"bertopic_model_{version_id[:8]}"
    if not model_path.exists():
        model_path = Path(__file__).parent.parent.parent / "models" / "bertopic_model"

    if model_path.exists():
        try:
            return BERTopic.load(str(model_path), embedding_model=embedding_model)
        except Exception as e:
            st.warning(f"Could not load BERTopic model from filesystem: {e}")
            return None

    # Model not found anywhere
    st.info("BERTopic model not found. Run the pipeline to generate visualizations.")
    return None


@st.cache_data(ttl=300)
def load_entity_statistics(version_id=None, entity_type=None, limit=100):
    """Load entity statistics for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        return db.get_entity_statistics(
            result_version_id=version_id,
            entity_type=entity_type,
            limit=limit
        )


@st.cache_data(ttl=300)
def load_summaries(version_id=None, source_id=None, limit=100):
    """Load article summaries with article metadata for a specific version."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if source_id:
                cur.execute(f"""
                    SELECT
                        s.id,
                        s.article_id,
                        s.summary_text,
                        s.method,
                        s.summary_length,
                        s.sentence_count,
                        s.word_count,
                        s.compression_ratio,
                        s.processing_time_ms,
                        s.created_at,
                        a.title,
                        a.content,
                        a.source_id,
                        a.date_posted,
                        a.url,
                        LENGTH(a.content) as original_length
                    FROM {schema}.article_summaries s
                    JOIN {schema}.news_articles a ON s.article_id = a.id
                    WHERE s.result_version_id = %s AND a.source_id = %s
                    ORDER BY a.id
                    LIMIT %s
                """, (version_id, source_id, limit))
            else:
                cur.execute(f"""
                    SELECT
                        s.id,
                        s.article_id,
                        s.summary_text,
                        s.method,
                        s.summary_length,
                        s.sentence_count,
                        s.word_count,
                        s.compression_ratio,
                        s.processing_time_ms,
                        s.created_at,
                        a.title,
                        a.content,
                        a.source_id,
                        a.date_posted,
                        a.url,
                        LENGTH(a.content) as original_length
                    FROM {schema}.article_summaries s
                    JOIN {schema}.news_articles a ON s.article_id = a.id
                    WHERE s.result_version_id = %s
                    ORDER BY a.id
                    LIMIT %s
                """, (version_id, limit))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_summary_statistics(version_id=None):
    """Load aggregate statistics for summaries."""
    if not version_id:
        return {}

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Overall statistics
            cur.execute(f"""
                SELECT
                    COUNT(*) as total_summaries,
                    AVG(compression_ratio) as avg_compression,
                    AVG(processing_time_ms) as avg_time_ms,
                    AVG(word_count) as avg_word_count,
                    MIN(word_count) as min_word_count,
                    MAX(word_count) as max_word_count
                FROM {schema}.article_summaries
                WHERE result_version_id = %s
            """, (version_id,))
            overall = cur.fetchone()

            # Statistics by source
            cur.execute(f"""
                SELECT
                    a.source_id,
                    COUNT(*) as count,
                    AVG(s.compression_ratio) as avg_compression,
                    AVG(s.processing_time_ms) as avg_time_ms,
                    AVG(s.word_count) as avg_word_count
                FROM {schema}.article_summaries s
                JOIN {schema}.news_articles a ON s.article_id = a.id
                WHERE s.result_version_id = %s
                GROUP BY a.source_id
                ORDER BY a.source_id
            """, (version_id,))
            by_source = cur.fetchall()

            return {
                "overall": overall,
                "by_source": by_source
            }


@st.cache_data(ttl=300)
def load_summaries_by_source(version_id=None):
    """Load summary statistics grouped by source."""
    if not version_id:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    a.source_id,
                    COUNT(*) as count,
                    AVG(s.compression_ratio) as avg_compression,
                    AVG(s.processing_time_ms) as avg_time_ms,
                    AVG(s.word_count) as avg_word_count,
                    AVG(s.sentence_count) as avg_sentence_count
                FROM {schema}.article_summaries s
                JOIN {schema}.news_articles a ON s.article_id = a.id
                WHERE s.result_version_id = %s
                GROUP BY a.source_id
                ORDER BY a.source_id
            """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_articles_by_topic(version_id=None, topic_name=None):
    """Load articles for a specific topic in a version."""
    if not version_id or not topic_name:
        return []

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.id,
                    n.title,
                    n.source_id,
                    n.date_posted,
                    n.url
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                JOIN {schema}.news_articles n ON aa.article_id = n.id
                WHERE t.name = %s
                  AND aa.result_version_id = %s
                  AND t.result_version_id = %s
                ORDER BY n.date_posted DESC
            """, (topic_name, version_id, version_id))
            return cur.fetchall()


# ============================================================================
# Ditwah Claims - Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_ditwah_claims(version_id, keyword=None):
    """Load claims, optionally filtered by keyword."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if keyword:
                keyword_pattern = f"%{keyword.lower()}%"
                cur.execute(f"""
                    SELECT * FROM {schema}.ditwah_claims
                    WHERE result_version_id = %s
                      AND LOWER(claim_text) LIKE %s
                    ORDER BY claim_order, article_count DESC
                    LIMIT 50
                """, (version_id, keyword_pattern))
            else:
                cur.execute(f"""
                    SELECT * FROM {schema}.ditwah_claims
                    WHERE result_version_id = %s
                    ORDER BY claim_order, article_count DESC
                """, (version_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_sentiment_by_source(claim_id):
    """Get average sentiment by source for a claim."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.source_id,
                    AVG(cs.sentiment_score) as avg_sentiment,
                    STDDEV(cs.sentiment_score) as stddev_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.claim_sentiment cs
                WHERE cs.claim_id = %s
                GROUP BY cs.source_id
                ORDER BY avg_sentiment DESC
            """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_stance_by_source(claim_id):
    """Get average stance by source for a claim."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as stddev_stance,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(*) as article_count
                FROM {schema}.claim_stance cs
                WHERE cs.claim_id = %s
                GROUP BY cs.source_id
                ORDER BY avg_stance DESC
            """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_sentiment_breakdown(claim_id):
    """Get sentiment distribution (very negative to very positive percentages) by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    source_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN sentiment_score <= -3 THEN 1 ELSE 0 END)::int as very_negative_count,
                    SUM(CASE WHEN sentiment_score <= -3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as very_negative_pct,
                    SUM(CASE WHEN sentiment_score > -3 AND sentiment_score <= -1 THEN 1 ELSE 0 END)::int as negative_count,
                    SUM(CASE WHEN sentiment_score > -3 AND sentiment_score <= -1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as negative_pct,
                    SUM(CASE WHEN sentiment_score > -1 AND sentiment_score < 1 THEN 1 ELSE 0 END)::int as neutral_count,
                    SUM(CASE WHEN sentiment_score > -1 AND sentiment_score < 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                    SUM(CASE WHEN sentiment_score >= 1 AND sentiment_score < 3 THEN 1 ELSE 0 END)::int as positive_count,
                    SUM(CASE WHEN sentiment_score >= 1 AND sentiment_score < 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as positive_pct,
                    SUM(CASE WHEN sentiment_score >= 3 THEN 1 ELSE 0 END)::int as very_positive_count,
                    SUM(CASE WHEN sentiment_score >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as very_positive_pct
                FROM {schema}.claim_sentiment
                WHERE claim_id = %s
                GROUP BY source_id
            """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_stance_breakdown(claim_id):
    """Get stance distribution (agree/neutral/disagree percentages) by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    source_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END)::int as agree_count,
                    SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as agree_pct,
                    SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END)::int as neutral_count,
                    SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                    SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END)::int as disagree_count,
                    SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as disagree_pct
                FROM {schema}.claim_stance
                WHERE claim_id = %s
                GROUP BY source_id
            """, (claim_id,))
            return cur.fetchall()


@st.cache_data(ttl=300)
def load_claim_articles(claim_id, limit=10):
    """Get sample articles for a claim with sentiment/stance scores."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.id,
                    n.title,
                    n.content,
                    n.date_posted,
                    n.url,
                    n.source_id,
                    cs_sentiment.sentiment_score,
                    cs_stance.stance_score,
                    cs_stance.stance_label,
                    cs_stance.supporting_quotes
                FROM {schema}.claim_sentiment cs_sentiment
                JOIN {schema}.claim_stance cs_stance
                    ON cs_sentiment.article_id = cs_stance.article_id
                    AND cs_sentiment.claim_id = cs_stance.claim_id
                JOIN {schema}.news_articles n ON n.id = cs_sentiment.article_id
                WHERE cs_sentiment.claim_id = %s
                ORDER BY n.date_posted DESC
                LIMIT %s
            """, (claim_id, limit))
            return cur.fetchall()


# ============================================================================
# Stance Distribution - Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_stance_overview(version_id):
    """Get high-level stance statistics."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Total claims with stance data
            cur.execute(f"""
                SELECT COUNT(DISTINCT claim_id) as total_claims
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
            """, (version_id,))
            total_claims = cur.fetchone()['total_claims']

            # Most controversial claim (highest stddev in stance_score)
            cur.execute(f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    STDDEV(cs.stance_score) as controversy
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text
                ORDER BY controversy DESC
                LIMIT 1
            """, (version_id,))
            most_controversial = cur.fetchone()

            # Strongest consensus claim (lowest stddev)
            cur.execute(f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as controversy
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text
                HAVING COUNT(DISTINCT cs.source_id) >= 2
                ORDER BY controversy ASC
                LIMIT 1
            """, (version_id,))
            strongest_consensus = cur.fetchone()

            # Average confidence
            cur.execute(f"""
                SELECT AVG(cs.confidence) as avg_confidence
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
            """, (version_id,))
            avg_confidence = cur.fetchone()['avg_confidence']

            return {
                'total_claims': total_claims,
                'most_controversial': most_controversial,
                'strongest_consensus': strongest_consensus,
                'avg_confidence': avg_confidence
            }


@st.cache_data(ttl=300)
def load_stance_polarization_matrix(version_id, category_filter=None):
    """Get claim × source heatmap data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            category_clause = "AND dc.category = %s" if category_filter else ""
            params = [version_id, category_filter] if category_filter else [version_id]

            cur.execute(f"""
                SELECT
                    dc.id as claim_id,
                    dc.claim_text,
                    dc.category,
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    AVG(cs.confidence) as avg_confidence,
                    STDDEV(cs.stance_score) OVER (PARTITION BY dc.id) as controversy_index,
                    COUNT(cs.id) as article_count
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s {category_clause}
                GROUP BY dc.id, dc.claim_text, dc.category, cs.source_id
                ORDER BY controversy_index DESC, dc.claim_text, cs.source_id
            """, params)

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_source_alignment_matrix(version_id):
    """Calculate source-to-source alignment scores."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                WITH source_stances AS (
                    SELECT
                        cs.claim_id,
                        cs.source_id,
                        CASE
                            WHEN cs.stance_score > 0.2 THEN 'agree'
                            WHEN cs.stance_score < -0.2 THEN 'disagree'
                            ELSE 'neutral'
                        END as stance_category
                    FROM {schema}.claim_stance cs
                    JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                    WHERE dc.result_version_id = %s
                )
                SELECT
                    s1.source_id as source1,
                    s2.source_id as source2,
                    COUNT(*) as total_claims,
                    SUM(CASE WHEN s1.stance_category = s2.stance_category THEN 1 ELSE 0 END) as agree_count,
                    SUM(CASE WHEN s1.stance_category != s2.stance_category THEN 1 ELSE 0 END) as disagree_count,
                    ROUND(SUM(CASE WHEN s1.stance_category = s2.stance_category THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as alignment_pct
                FROM source_stances s1
                JOIN source_stances s2 ON s1.claim_id = s2.claim_id AND s1.source_id < s2.source_id
                GROUP BY s1.source_id, s2.source_id
                ORDER BY alignment_pct DESC
            """, (version_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_confidence_weighted_stances(version_id):
    """Get bubble chart data."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    dc.id as claim_id,
                    dc.claim_text,
                    dc.category,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as stddev_stance,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(DISTINCT cs.article_id) as article_count,
                    COUNT(DISTINCT cs.source_id) as source_count
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text, dc.category
                HAVING COUNT(DISTINCT cs.article_id) >= 2
                ORDER BY stddev_stance DESC
            """, (version_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_claim_source_comparison(claim_id):
    """Get detailed comparison for a single claim across all sources."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    cs.stance_label,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(cs.article_id) as article_count,
                    (ARRAY_AGG(cs.supporting_quotes ORDER BY cs.processed_at DESC))[1] as sample_quotes
                FROM {schema}.claim_stance cs
                WHERE cs.claim_id = %s
                GROUP BY cs.source_id, cs.stance_label
                ORDER BY avg_stance DESC
            """, (claim_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_claim_quotes_by_stance(claim_id):
    """Get supporting quotes grouped by stance."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.stance_label,
                    cs.stance_score,
                    cs.supporting_quotes,
                    cs.source_id,
                    n.title as article_title,
                    n.id as article_id,
                    n.date_posted
                FROM {schema}.claim_stance cs
                JOIN {schema}.news_articles n ON cs.article_id = n.id
                WHERE cs.claim_id = %s
                  AND cs.supporting_quotes IS NOT NULL
                  AND jsonb_array_length(cs.supporting_quotes) > 0
                ORDER BY cs.stance_score DESC, n.date_posted DESC
            """, (claim_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_stance_by_category(version_id):
    """Get stance patterns grouped by claim category."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    dc.category,
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    COUNT(DISTINCT dc.id) as claim_count,
                    COUNT(cs.article_id) as article_count
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.category, cs.source_id
                ORDER BY dc.category, cs.source_id
            """, (version_id,))

            rows = cur.fetchall()
            return pd.DataFrame(rows)


# ============================================================================
# Article Insights loaders
# ============================================================================

@st.cache_data(ttl=60)
def search_articles_by_title(search_term: str, limit: int = 50):
    """Search articles by title using LIKE query.

    Returns:
        List of dicts with columns [id, title, source_id, date_posted]
    """
    if not search_term or len(search_term) < 2:
        return []

    with get_db() as db:
        return db.search_articles(search_term, limit)


@st.cache_data(ttl=300)
def load_article_by_id(article_id: int):
    """Load article metadata.

    Returns:
        Dict with {id, title, content, source_id, date_posted, url, lang, is_ditwah_cyclone}
    """
    with get_db() as db:
        return db.get_article_by_id(article_id)


@st.cache_data(ttl=300)
def load_article_sentiment(article_id: int, model_type: str = 'roberta'):
    """Load sentiment analysis for article.

    Returns:
        Dict with {overall_sentiment, headline_sentiment, confidence, reasoning, model_type}
    """
    with get_db() as db:
        return db.get_sentiment_for_article(article_id, model_type)


@st.cache_data(ttl=300)
def load_article_topic(article_id: int, version_id: str):
    """Load topic assignment for article.

    Returns:
        Dict with {topic_id, topic_name, topic_confidence}
    """
    with get_db() as db:
        return db.get_topic_for_article(article_id, version_id)


@st.cache_data(ttl=300)
def load_article_summary(article_id: int, version_id: str):
    """Load summary for article.

    Returns:
        Dict with {summary_text, method, compression_ratio, word_count, summary_length, processing_time_ms}
    """
    with get_db() as db:
        return db.get_summary_for_article(article_id, version_id)


@st.cache_data(ttl=300)
def load_article_entities(article_id: int, version_id: str):
    """Load named entities for article.

    Returns:
        List of dicts with [entity_text, entity_type, confidence, start_char, end_char]
    """
    with get_db() as db:
        return db.get_entities_for_article(article_id, version_id)


@st.cache_data(ttl=300)
def load_article_cluster(article_id: int, version_id: str):
    """Load event cluster assignment.

    Returns:
        Dict with {cluster_id, cluster_name, similarity_score, other_sources[], article_count}
    """
    with get_db() as db:
        return db.get_cluster_for_article(article_id, version_id)


@st.cache_data(ttl=300)
def get_available_sentiment_models():
    """Get list of sentiment models that have analyzed articles.

    Returns:
        List of model types (e.g., ['roberta', 'vader', 'distilbert'])
    """
    models = load_available_models()
    return [m['model_type'] for m in models] if models else []


@st.cache_data(ttl=300)
def load_article_claims(article_id: str):
    """Load claims linked to a specific article.

    Args:
        article_id: The article UUID (as string)

    Returns:
        List of dicts with claim info and stance/sentiment for this article
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    dc.id as claim_id,
                    dc.claim_text,
                    dc.claim_category,
                    dc.claim_order,
                    cs_stance.stance_score,
                    cs_stance.stance_label,
                    cs_stance.confidence,
                    cs_stance.supporting_quotes,
                    cs_sentiment.sentiment_score
                FROM {schema}.ditwah_claims dc
                LEFT JOIN {schema}.claim_stance cs_stance
                    ON dc.id = cs_stance.claim_id
                    AND cs_stance.article_id = %s::uuid
                LEFT JOIN {schema}.claim_sentiment cs_sentiment
                    ON dc.id = cs_sentiment.claim_id
                    AND cs_sentiment.article_id = %s::uuid
                WHERE cs_stance.article_id = %s::uuid OR cs_sentiment.article_id = %s::uuid
                ORDER BY dc.claim_order, dc.claim_text
            """, (article_id, article_id, article_id, article_id))
            return cur.fetchall()


def load_multi_doc_summary_for_topic(article_id: int, topic_version_id: str, multi_doc_version_id: str):
    """Load or generate multi-document summary for article's topic.

    Args:
        article_id: Article ID
        topic_version_id: Topic analysis version ID
        multi_doc_version_id: Multi-doc summarization version ID

    Returns:
        Dict with {summary_text, article_count, source_count, word_count, processing_time_ms, generated_now}
        or Dict with {error: 'not_enough_articles', article_count: N} if < 2 articles
        or Dict with {error: 'generation_failed'} if LLM fails
        or None if article has no topic assignment
    """
    with get_db() as db:
        # 1. Get article's topic assignment
        topic = db.get_topic_for_article(article_id, topic_version_id)
        if not topic:
            return None

        topic_id = topic['topic_id']

        # 2. Check if summary already exists
        existing = db.get_multi_doc_summary('topic', str(topic_id), multi_doc_version_id, topic_version_id)
        if existing:
            return {
                'summary_text': existing['summary_text'],
                'article_count': existing['article_count'],
                'source_count': existing['source_count'],
                'word_count': existing['word_count'],
                'processing_time_ms': existing['processing_time_ms'],
                'method': existing['method'],
                'llm_model': existing['llm_model'],
                'generated_now': False
            }

        # 3. Generate new summary
        import time
        from src.multi_doc_summarization import OpenAIMultiDocSummarizer, GeminiMultiDocSummarizer
        from src.versions import get_version
        from dashboard.components.source_mapping import SOURCE_NAMES

        # Get version configuration
        version = get_version(multi_doc_version_id)
        if not version:
            return None

        mds_config = version['configuration'].get('multi_doc_summarization', {})
        method = mds_config.get('method', 'gemini')
        llm_model = mds_config.get('llm_model', 'gemini-2.0-flash')

        # Get all articles in topic
        all_articles = db.get_articles_by_topic(topic_id, topic_version_id)

        if len(all_articles) < 2:
            # Need at least 2 articles for multi-doc summary
            return {'error': 'not_enough_articles', 'article_count': len(all_articles)}

        # Smart article sampling if needed
        max_articles = mds_config.get('max_articles', 10 if method == 'openai' else 50)
        sampled_from = len(all_articles)

        if len(all_articles) > max_articles:
            # Sample articles with source diversity: take most recent from each source
            from collections import defaultdict
            articles_by_source = defaultdict(list)
            for article in all_articles:
                articles_by_source[article['source_id']].append(article)

            # Calculate how many to take from each source
            num_sources = len(articles_by_source)
            per_source = max(1, max_articles // num_sources)

            # Take most recent articles from each source
            sampled_articles = []
            for source_articles in articles_by_source.values():
                # Already sorted by date in get_articles_by_topic
                sampled_articles.extend(source_articles[:per_source])

            # If we're under max_articles, add more from largest sources
            if len(sampled_articles) < max_articles:
                remaining = max_articles - len(sampled_articles)
                for source_articles in sorted(articles_by_source.values(), key=len, reverse=True):
                    if remaining <= 0:
                        break
                    additional = source_articles[per_source:per_source + remaining]
                    sampled_articles.extend(additional)
                    remaining -= len(additional)

            articles = sampled_articles[:max_articles]
        else:
            articles = all_articles

        # Prepare documents and sources
        documents = [a['content'] for a in articles]
        sources = [SOURCE_NAMES.get(a['source_id'], a['source_id']) for a in articles]

        # Initialize summarizer
        summarizer_config = {
            'method': method,
            'llm_model': llm_model,
            'llm_temperature': mds_config.get('temperature', 0.0),
            'summary_length': mds_config.get('summary_length', 'medium'),
            'short_sentences': mds_config.get('short_sentences', 5),
            'short_words': mds_config.get('short_words', 80),
            'medium_sentences': mds_config.get('medium_sentences', 8),
            'medium_words': mds_config.get('medium_words', 150),
            'long_sentences': mds_config.get('long_sentences', 12),
            'long_words': mds_config.get('long_words', 200)
        }

        if method == 'openai':
            summarizer = OpenAIMultiDocSummarizer(summarizer_config)
        else:
            summarizer = GeminiMultiDocSummarizer(summarizer_config)

        # Generate summary with error handling
        start_time = time.time()
        try:
            summary_text = summarizer.summarize_multiple(documents, sources)
            processing_time_ms = int((time.time() - start_time) * 1000)

            if not summary_text:
                return {
                    'error': 'generation_failed',
                    'article_count': len(articles),
                    'error_message': 'LLM returned empty response',
                    'sampled': len(articles) < sampled_from,
                    'sampled_from': sampled_from
                }
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_str = str(e)

            # Parse common API errors for user-friendly messages
            if 'rate_limit_exceeded' in error_str or '429' in error_str:
                error_type = 'rate_limit'
                if 'tokens per min' in error_str.lower() or 'tpm' in error_str.lower():
                    error_message = 'Token rate limit exceeded. The articles contain too many tokens for your API plan. Try reducing max_articles or upgrading your API plan.'
                else:
                    error_message = 'API rate limit exceeded. Please wait a moment and try again.'
            elif 'quota' in error_str.lower():
                error_type = 'quota_exceeded'
                error_message = 'API quota exceeded. Check your billing and usage limits.'
            elif 'authentication' in error_str.lower() or '401' in error_str or '403' in error_str:
                error_type = 'authentication'
                error_message = 'API authentication failed. Check your API key configuration.'
            elif 'timeout' in error_str.lower():
                error_type = 'timeout'
                error_message = 'Request timed out. The content may be too large. Try reducing max_articles.'
            else:
                error_type = 'api_error'
                error_message = f'API error: {error_str[:200]}'  # Truncate long errors

            return {
                'error': 'generation_failed',
                'error_type': error_type,
                'error_message': error_message,
                'article_count': len(articles),
                'sampled': len(articles) < sampled_from,
                'sampled_from': sampled_from,
                'full_error': error_str  # For debugging
            }

        # Count words
        word_count = len(summary_text.split())

        # Count unique sources
        source_count = len(set(a['source_id'] for a in articles))

        # Store in database
        db.store_multi_doc_summary(
            group_type='topic',
            group_id=str(topic_id),
            version_id=multi_doc_version_id,
            source_version_id=topic_version_id,
            summary_text=summary_text,
            method=method,
            llm_model=llm_model,
            article_count=len(articles),
            source_count=source_count,
            word_count=word_count,
            processing_time_ms=processing_time_ms
        )

        return {
            'summary_text': summary_text,
            'article_count': len(articles),
            'source_count': source_count,
            'word_count': word_count,
            'processing_time_ms': processing_time_ms,
            'method': method,
            'llm_model': llm_model,
            'generated_now': True,
            'sampled': len(articles) < sampled_from,
            'sampled_from': sampled_from
        }


def load_multi_doc_summary_for_cluster(article_id: int, cluster_version_id: str, multi_doc_version_id: str):
    """Load or generate multi-document summary for article's event cluster.

    Args:
        article_id: Article ID
        cluster_version_id: Clustering analysis version ID
        multi_doc_version_id: Multi-doc summarization version ID

    Returns:
        Dict with {summary_text, article_count, source_count, word_count, processing_time_ms, generated_now}
        or Dict with {error: 'not_enough_articles', article_count: N} if < 2 articles
        or Dict with {error: 'generation_failed'} if LLM fails
        or None if article has no cluster assignment
    """
    with get_db() as db:
        # 1. Get article's cluster assignment
        cluster = db.get_cluster_for_article(article_id, cluster_version_id)
        if not cluster:
            return None

        cluster_id = cluster['cluster_id']

        # 2. Check if summary already exists
        existing = db.get_multi_doc_summary('cluster', cluster_id, multi_doc_version_id, cluster_version_id)
        if existing:
            return {
                'summary_text': existing['summary_text'],
                'article_count': existing['article_count'],
                'source_count': existing['source_count'],
                'word_count': existing['word_count'],
                'processing_time_ms': existing['processing_time_ms'],
                'method': existing['method'],
                'llm_model': existing['llm_model'],
                'generated_now': False
            }

        # 3. Generate new summary
        import time
        from src.multi_doc_summarization import OpenAIMultiDocSummarizer, GeminiMultiDocSummarizer
        from src.versions import get_version
        from dashboard.components.source_mapping import SOURCE_NAMES

        # Get version configuration
        version = get_version(multi_doc_version_id)
        if not version:
            return None

        mds_config = version['configuration'].get('multi_doc_summarization', {})
        method = mds_config.get('method', 'gemini')
        llm_model = mds_config.get('llm_model', 'gemini-2.0-flash')

        # Get all articles in cluster
        all_articles = db.get_articles_by_cluster(cluster_id, cluster_version_id)

        if len(all_articles) < 2:
            # Need at least 2 articles for multi-doc summary
            return {'error': 'not_enough_articles', 'article_count': len(all_articles)}

        # Smart article sampling if needed
        max_articles = mds_config.get('max_articles', 10 if method == 'openai' else 50)
        sampled_from = len(all_articles)

        if len(all_articles) > max_articles:
            # Sample articles with source diversity: take most recent from each source
            from collections import defaultdict
            articles_by_source = defaultdict(list)
            for article in all_articles:
                articles_by_source[article['source_id']].append(article)

            # Calculate how many to take from each source
            num_sources = len(articles_by_source)
            per_source = max(1, max_articles // num_sources)

            # Take most recent articles from each source
            sampled_articles = []
            for source_articles in articles_by_source.values():
                # Already sorted by similarity in get_articles_by_cluster
                sampled_articles.extend(source_articles[:per_source])

            # If we're under max_articles, add more from largest sources
            if len(sampled_articles) < max_articles:
                remaining = max_articles - len(sampled_articles)
                for source_articles in sorted(articles_by_source.values(), key=len, reverse=True):
                    if remaining <= 0:
                        break
                    additional = source_articles[per_source:per_source + remaining]
                    sampled_articles.extend(additional)
                    remaining -= len(additional)

            articles = sampled_articles[:max_articles]
        else:
            articles = all_articles

        # Prepare documents and sources
        documents = [a['content'] for a in articles]
        sources = [SOURCE_NAMES.get(a['source_id'], a['source_id']) for a in articles]

        # Initialize summarizer
        summarizer_config = {
            'method': method,
            'llm_model': llm_model,
            'llm_temperature': mds_config.get('temperature', 0.0),
            'summary_length': mds_config.get('summary_length', 'medium'),
            'short_sentences': mds_config.get('short_sentences', 5),
            'short_words': mds_config.get('short_words', 80),
            'medium_sentences': mds_config.get('medium_sentences', 8),
            'medium_words': mds_config.get('medium_words', 150),
            'long_sentences': mds_config.get('long_sentences', 12),
            'long_words': mds_config.get('long_words', 200)
        }

        if method == 'openai':
            summarizer = OpenAIMultiDocSummarizer(summarizer_config)
        else:
            summarizer = GeminiMultiDocSummarizer(summarizer_config)

        # Generate summary with error handling
        start_time = time.time()
        try:
            summary_text = summarizer.summarize_multiple(documents, sources)
            processing_time_ms = int((time.time() - start_time) * 1000)

            if not summary_text:
                return {
                    'error': 'generation_failed',
                    'article_count': len(articles),
                    'error_message': 'LLM returned empty response',
                    'sampled': len(articles) < sampled_from,
                    'sampled_from': sampled_from
                }
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            error_str = str(e)

            # Parse common API errors for user-friendly messages
            if 'rate_limit_exceeded' in error_str or '429' in error_str:
                error_type = 'rate_limit'
                if 'tokens per min' in error_str.lower() or 'tpm' in error_str.lower():
                    error_message = 'Token rate limit exceeded. The articles contain too many tokens for your API plan. Try reducing max_articles or upgrading your API plan.'
                else:
                    error_message = 'API rate limit exceeded. Please wait a moment and try again.'
            elif 'quota' in error_str.lower():
                error_type = 'quota_exceeded'
                error_message = 'API quota exceeded. Check your billing and usage limits.'
            elif 'authentication' in error_str.lower() or '401' in error_str or '403' in error_str:
                error_type = 'authentication'
                error_message = 'API authentication failed. Check your API key configuration.'
            elif 'timeout' in error_str.lower():
                error_type = 'timeout'
                error_message = 'Request timed out. The content may be too large. Try reducing max_articles.'
            else:
                error_type = 'api_error'
                error_message = f'API error: {error_str[:200]}'  # Truncate long errors

            return {
                'error': 'generation_failed',
                'error_type': error_type,
                'error_message': error_message,
                'article_count': len(articles),
                'sampled': len(articles) < sampled_from,
                'sampled_from': sampled_from,
                'full_error': error_str  # For debugging
            }

        # Count words
        word_count = len(summary_text.split())

        # Count unique sources
        source_count = len(set(a['source_id'] for a in articles))

        # Store in database
        db.store_multi_doc_summary(
            group_type='cluster',
            group_id=cluster_id,
            version_id=multi_doc_version_id,
            source_version_id=cluster_version_id,
            summary_text=summary_text,
            method=method,
            llm_model=llm_model,
            article_count=len(articles),
            source_count=source_count,
            word_count=word_count,
            processing_time_ms=processing_time_ms
        )

        return {
            'summary_text': summary_text,
            'article_count': len(articles),
            'source_count': source_count,
            'word_count': word_count,
            'processing_time_ms': processing_time_ms,
            'method': method,
            'llm_model': llm_model,
            'generated_now': True,
            'sampled': len(articles) < sampled_from,
            'sampled_from': sampled_from
        }
