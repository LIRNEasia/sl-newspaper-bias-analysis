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


# Article Insights loaders
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
        Dict with {topic_id, topic_name, confidence, overall_tone, headline_tone}
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
