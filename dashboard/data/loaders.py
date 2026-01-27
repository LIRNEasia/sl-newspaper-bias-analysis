"""Data loading functions with Streamlit caching."""

import streamlit as st
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

            # Articles by source
            cur.execute(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.news_articles
                GROUP BY source_id
                ORDER BY count DESC
            """)
            by_source = cur.fetchall()

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
        "by_source": by_source,
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

    # Strategy 1: Try loading from database
    from src.versions import get_model_from_version
    import tempfile

    try:
        # Extract model from database to temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"bertopic_{version_id[:8]}_")
        model_path = get_model_from_version(version_id, temp_dir)

        if model_path:
            try:
                model = BERTopic.load(model_path)
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
            return BERTopic.load(str(model_path))
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
