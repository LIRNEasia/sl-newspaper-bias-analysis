"""Database connection and operations for media bias analysis."""

import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from typing import List, Dict, Any
from contextlib import contextmanager

from src.config import load_config


class Database:
    """PostgreSQL database connection manager."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()["database"]
        self._conn = None

    def connect(self):
        """Establish database connection."""
        self._conn = psycopg2.connect(
            host=self.config["host"],
            port=self.config["port"],
            dbname=self.config["name"],
            user=self.config["user"],
            password=self.config["password"]
        )
        self._conn.autocommit = False
        return self

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self, dict_cursor: bool = True):
        """Context manager for database cursor."""
        cursor_factory = RealDictCursor if dict_cursor else None
        cur = self._conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Article operations
    def get_articles(
        self,
        limit: int = None,
        offset: int = 0,
        source_id: str = None
    ) -> List[Dict]:
        """Fetch articles from news_articles table."""
        schema = self.config["schema"]
        query = f"""
            SELECT id, url, title, content, date_posted, source_id, lang
            FROM {schema}.news_articles
            WHERE content IS NOT NULL AND content != '' AND is_ditwah_cyclone = 1
        """
        params = []

        if source_id:
            query += " AND source_id = %s"
            params.append(source_id)

        query += " ORDER BY date_posted, id"

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_article_count(self) -> int:
        """Get total article count."""
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.news_articles
                WHERE content IS NOT NULL AND content != '' AND is_ditwah_cyclone = 1
            """)
            return cur.fetchone()["count"]

    def get_article_by_url(self, url: str) -> Dict:
        """Fetch article by URL.

        Args:
            url: The article URL to search for

        Returns:
            Article dict with id, url, title, content, source_id, date_posted, or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, url, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE url = %s AND is_ditwah_cyclone = 1
            """, (url,))
            return cur.fetchone()

    def get_articles_without_embeddings(self, result_version_id: str = None, limit: int = None) -> List[Dict]:
        """Get articles that don't have embeddings yet for a specific version."""
        schema = self.config["schema"]

        if result_version_id:
            query = f"""
                SELECT a.id, a.title, a.content, a.date_posted, a.source_id
                FROM {schema}.news_articles a
                LEFT JOIN {schema}.embeddings e ON a.id = e.article_id AND e.result_version_id = %s
                WHERE e.id IS NULL
                  AND a.content IS NOT NULL
                  AND a.content != ''
                  AND a.is_ditwah_cyclone = 1
                ORDER BY a.date_posted, a.id
            """
            params = [result_version_id]
        else:
            query = f"""
                SELECT a.id, a.title, a.content, a.date_posted, a.source_id
                FROM {schema}.news_articles a
                LEFT JOIN {schema}.embeddings e ON a.id = e.article_id
                WHERE e.id IS NULL
                  AND a.content IS NOT NULL
                  AND a.content != ''
                  AND is_ditwah_cyclone = 1
                ORDER BY a.date_posted, a.id
            """
            params = []

        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    # Embedding operations
    def store_embeddings(self, embeddings: List[Dict[str, Any]], result_version_id: str):
        """Store article embeddings in batch.

        Args:
            embeddings: List of dicts with 'article_id' and 'embedding' keys
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.embeddings (article_id, result_version_id, embedding, embedding_model)
                VALUES %s
                ON CONFLICT (article_id, result_version_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    created_at = NOW()
                """,
                [
                    (e["article_id"], result_version_id, e["embedding"], e.get("model", "all-mpnet-base-v2"))
                    for e in embeddings
                ],
                template="(%s, %s, %s::vector, %s)"
            )

    def get_all_embeddings(self, result_version_id: str = None) -> List[Dict]:
        """Get all article embeddings for a specific version."""
        schema = self.config["schema"]

        if result_version_id:
            query = f"""
                SELECT e.article_id, e.embedding::text, a.title, a.content,
                       a.date_posted, a.source_id
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles a ON e.article_id = a.id
                WHERE e.result_version_id = %s
                  AND a.is_ditwah_cyclone = 1
                ORDER BY a.date_posted, a.id
            """
            params = [result_version_id]
        else:
            query = f"""
                SELECT e.article_id, e.embedding::text, a.title, a.content,
                       a.date_posted, a.source_id
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles a ON e.article_id = a.id
                WHERE a.is_ditwah_cyclone = 1
                ORDER BY a.date_posted, a.id
            """
            params = []

        with self.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            # Parse embedding strings to float arrays
            result = []
            for row in rows:
                embedding_str = row['embedding']
                # Parse the pgvector format: [0.1,0.2,0.3,...]
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding = [float(x) for x in embedding_str[1:-1].split(',')]
                else:
                    embedding = [float(x) for x in embedding_str.split(',')]

                result.append({
                    'article_id': row['article_id'],
                    'embedding': embedding,
                    'title': row['title'],
                    'content': row['content'],
                    'date_posted': row['date_posted'],
                    'source_id': row['source_id']
                })
            return result

    def get_embedding_count(self, result_version_id: str = None) -> int:
        """Get count of articles with embeddings for a specific version."""
        schema = self.config["schema"]
        with self.cursor() as cur:
            if result_version_id:
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.embeddings WHERE result_version_id = %s",
                    (result_version_id,)
                )
            else:
                cur.execute(f"SELECT COUNT(*) as count FROM {schema}.embeddings")
            return cur.fetchone()["count"]

    # Topic operations
    def store_topics(self, topics: List[Dict], result_version_id: str):
        """Store discovered topics for a specific version."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            for topic in topics:
                cur.execute(f"""
                    INSERT INTO {schema}.topics
                    (topic_id, result_version_id, parent_topic_id, name, description, keywords, article_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (topic_id, result_version_id) DO UPDATE SET
                        parent_topic_id = EXCLUDED.parent_topic_id,
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        article_count = EXCLUDED.article_count
                """, (
                    topic["topic_id"],
                    result_version_id,
                    topic.get("parent_topic_id"),
                    topic["name"],
                    topic.get("description"),
                    topic.get("keywords", []),
                    topic.get("article_count", 0)
                ))

    def store_article_topics(self, assignments: List[Dict], result_version_id: str):
        """Store topic assignments for articles for a specific version."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            # First, build a mapping from BERTopic topic_id to database id
            cur.execute(
                f"""
                SELECT id, topic_id FROM {schema}.topics
                WHERE result_version_id = %s
                """,
                (result_version_id,)
            )
            topic_id_to_db_id = {row[1]: row[0] for row in cur.fetchall()}

            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.article_analysis
                (article_id, result_version_id, primary_topic_id, topic_confidence)
                VALUES %s
                ON CONFLICT (article_id, result_version_id) DO UPDATE SET
                    primary_topic_id = EXCLUDED.primary_topic_id,
                    topic_confidence = EXCLUDED.topic_confidence
                """,
                [(a["article_id"], result_version_id, topic_id_to_db_id.get(a["topic_id"]), a.get("confidence", 0.0))
                 for a in assignments]
            )

    # Analysis operations
    def store_article_analysis(self, analyses: List[Dict]):
        """Store article analysis results (tone, type)."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            for a in analyses:
                cur.execute(f"""
                    INSERT INTO {schema}.article_analysis
                    (article_id, article_type, article_type_confidence,
                     overall_tone, headline_tone, tone_reasoning,
                     llm_provider, llm_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET
                        article_type = EXCLUDED.article_type,
                        article_type_confidence = EXCLUDED.article_type_confidence,
                        overall_tone = EXCLUDED.overall_tone,
                        headline_tone = EXCLUDED.headline_tone,
                        tone_reasoning = EXCLUDED.tone_reasoning,
                        llm_provider = EXCLUDED.llm_provider,
                        llm_model = EXCLUDED.llm_model,
                        processed_at = NOW()
                """, (
                    a["article_id"],
                    a.get("article_type"),
                    a.get("article_type_confidence"),
                    a.get("overall_tone"),
                    a.get("headline_tone"),
                    a.get("tone_reasoning"),
                    a.get("llm_provider"),
                    a.get("llm_model")
                ))

    def get_articles_without_analysis(self, limit: int = None) -> List[Dict]:
        """Get articles that haven't been analyzed yet."""
        schema = self.config["schema"]
        query = f"""
            SELECT a.id, a.title, a.content, a.date_posted, a.source_id
            FROM {schema}.news_articles a
            LEFT JOIN {schema}.article_analysis aa ON a.id = aa.article_id
            WHERE (aa.overall_tone IS NULL OR aa.article_type IS NULL)
              AND a.content IS NOT NULL
              AND a.content != ''
              AND is_ditwah_cyclone = 1
            ORDER BY a.date_posted, a.id
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    # Event cluster operations
    def store_event_clusters(self, clusters: List[Dict], result_version_id: str):
        """Store event clusters for a specific version."""
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            for cluster in clusters:
                cur.execute(f"""
                    INSERT INTO {schema}.event_clusters
                    (id, result_version_id, cluster_name, cluster_description, representative_article_id,
                     article_count, sources_count, date_start, date_end, centroid_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        result_version_id = EXCLUDED.result_version_id,
                        cluster_name = EXCLUDED.cluster_name,
                        cluster_description = EXCLUDED.cluster_description,
                        representative_article_id = EXCLUDED.representative_article_id,
                        article_count = EXCLUDED.article_count,
                        sources_count = EXCLUDED.sources_count,
                        date_start = EXCLUDED.date_start,
                        date_end = EXCLUDED.date_end,
                        centroid_embedding = EXCLUDED.centroid_embedding
                """, (
                    cluster["id"],
                    result_version_id,
                    cluster["name"],
                    cluster.get("description"),
                    cluster["representative_article_id"],
                    cluster["article_count"],
                    cluster["sources_count"],
                    cluster["date_start"],
                    cluster["date_end"],
                    cluster.get("centroid")
                ))

                # Store article-cluster mappings
                if cluster.get("articles"):
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {schema}.article_clusters
                        (article_id, cluster_id, result_version_id, similarity_score)
                        VALUES %s
                        ON CONFLICT (article_id, cluster_id, result_version_id) DO UPDATE SET
                            similarity_score = EXCLUDED.similarity_score
                        """,
                        [(a["article_id"], cluster["id"], result_version_id, a.get("similarity", 0.0))
                         for a in cluster["articles"]]
                    )
    # Word frequency operations
    def store_word_frequencies(self, frequencies: List[Dict], result_version_id: str):
        """Store word frequency results for a specific version.

        Args:
            frequencies: List of dicts with 'source_id', 'word', 'frequency', 'tfidf_score', 'rank'
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.word_frequencies
                (result_version_id, source_id, word, frequency, tfidf_score, rank)
                VALUES %s
                ON CONFLICT (result_version_id, source_id, word) DO UPDATE SET
                    frequency = EXCLUDED.frequency,
                    tfidf_score = EXCLUDED.tfidf_score,
                    rank = EXCLUDED.rank,
                    created_at = NOW()
                """,
                [
                    (
                        result_version_id,
                        f["source_id"],
                        f["word"],
                        f["frequency"],
                        f.get("tfidf_score"),
                        f["rank"]
                    )
                    for f in frequencies
                ]
            )

    def get_word_frequencies(
        self,
        result_version_id: str,
        source_id: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get word frequencies for a specific version and optional source.

        Args:
            result_version_id: UUID of the result version
            source_id: Optional source filter
            limit: Maximum number of words to return per source

        Returns:
            List of dicts with word frequency data
        """
        schema = self.config["schema"]
        params = [result_version_id]

        query = f"""
            SELECT source_id, word, frequency, tfidf_score, rank
            FROM {schema}.word_frequencies
            WHERE result_version_id = %s
        """

        if source_id:
            query += " AND source_id = %s"
            params.append(source_id)

        query += " AND rank <= %s ORDER BY source_id, rank"
        params.append(limit)

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    # Named entity operations
    def store_named_entities(
        self,
        entities: List[Dict[str, Any]],
        result_version_id: str
    ) -> None:
        """
        Store named entities in the database.

        Args:
            entities: List of entity dictionaries
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            for entity in entities:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.named_entities
                    (result_version_id, article_id, entity_text, entity_type,
                     start_char, end_char, confidence, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, result_version_id, entity_text, entity_type, start_char)
                    DO NOTHING
                    """,
                    (
                        result_version_id,
                        entity["article_id"],
                        entity["entity_text"],
                        entity["entity_type"],
                        entity["start_char"],
                        entity["end_char"],
                        entity["confidence"],
                        entity.get("context", "")
                    )
                )

    def compute_entity_statistics(self, result_version_id: str) -> None:
        """
        Compute aggregated entity statistics per source.

        Args:
            result_version_id: UUID of the result version
        """
        schema = self.config["schema"]

        with self.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.entity_statistics
                (result_version_id, entity_text, entity_type, source_id,
                 mention_count, article_count, avg_confidence)
                SELECT
                    ne.result_version_id,
                    ne.entity_text,
                    ne.entity_type,
                    na.source_id,
                    COUNT(*) as mention_count,
                    COUNT(DISTINCT ne.article_id) as article_count,
                    AVG(ne.confidence) as avg_confidence
                FROM {schema}.named_entities ne
                JOIN {schema}.news_articles na ON ne.article_id = na.id
                WHERE ne.result_version_id = %s
                  AND na.is_ditwah_cyclone = 1
                GROUP BY ne.result_version_id, ne.entity_text, ne.entity_type, na.source_id
                ON CONFLICT (result_version_id, entity_text, entity_type, source_id)
                DO UPDATE SET
                    mention_count = EXCLUDED.mention_count,
                    article_count = EXCLUDED.article_count,
                    avg_confidence = EXCLUDED.avg_confidence
                """,
                (result_version_id,)
            )

    def get_entity_statistics(
        self,
        result_version_id: str,
        entity_type: str = None,
        source_id: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get entity statistics for a version.

        Args:
            result_version_id: UUID of the result version
            entity_type: Optional filter by entity type
            source_id: Optional filter by source
            limit: Maximum number of results

        Returns:
            List of entity statistics
        """
        schema = self.config["schema"]

        query = f"""
            SELECT entity_text, entity_type, source_id,
                   mention_count, article_count, avg_confidence
            FROM {schema}.entity_statistics
            WHERE result_version_id = %s
        """
        params = [result_version_id]

        if entity_type:
            query += " AND entity_type = %s"
            params.append(entity_type)

        if source_id:
            query += " AND source_id = %s"
            params.append(source_id)

        query += " ORDER BY mention_count DESC LIMIT %s"
        params.append(limit)

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_entities_for_article(
        self,
        article_id: str,
        result_version_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all named entities for a specific article.

        Args:
            article_id: The article ID
            result_version_id: UUID of the result version

        Returns:
            List of entity dicts with entity_text, entity_type, start_char, end_char, confidence
            Ordered by start_char for sequential processing
        """
        schema = self.config["schema"]

        query = f"""
            SELECT entity_text, entity_type, start_char, end_char, confidence
            FROM {schema}.named_entities
            WHERE article_id = %s AND result_version_id = %s
            ORDER BY start_char
        """

        with self.cursor() as cur:
            cur.execute(query, (article_id, result_version_id))
            return cur.fetchall()

    def get_unique_entity_texts(
        self,
        result_version_id: str = None,
        entity_types: List[str] = None,
        normalize: bool = True
    ) -> List[str]:
        """
        Get unique entity texts from NER analysis for use as stop words.

        Args:
            result_version_id: Optional NER version ID. If None, uses any completed NER version.
            entity_types: Optional list of entity types to filter (e.g., ['PERSON', 'ORG', 'GPE'])
            normalize: If True, lowercase and deduplicate entities

        Returns:
            List of unique entity text strings
        """
        schema = self.config["schema"]

        # If no version specified, find first completed NER version
        if result_version_id is None:
            with self.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id FROM {schema}.result_versions
                    WHERE analysis_type = 'ner'
                      AND (pipeline_status->>'ner')::boolean = true
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("No completed NER analysis found. Please run NER pipeline first.")
                result_version_id = str(row["id"])

        # Build query to get unique entity texts
        query = f"""
            SELECT DISTINCT entity_text
            FROM {schema}.named_entities
            WHERE result_version_id = %s
        """
        params = [result_version_id]

        # Filter by entity types if provided
        if entity_types:
            query += " AND entity_type = ANY(%s)"
            params.append(entity_types)

        with self.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        # Extract entity texts
        entity_texts = [row["entity_text"] for row in rows]

        # Normalize if requested
        if normalize:
            # Lowercase and deduplicate
            entity_texts = list(set(text.lower() for text in entity_texts))

        return sorted(entity_texts)
    # Sentiment analysis operations
    def store_sentiment_analyses(self, analyses: List[Dict]):
        """Store sentiment analysis results in batch.

        Args:
            analyses: List of dicts with sentiment analysis results
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.sentiment_analyses
                (article_id, model_type, model_name, overall_sentiment, overall_confidence,
                 headline_sentiment, headline_confidence, sentiment_reasoning,
                 sentiment_aspects, processing_time_ms)
                VALUES %s
                ON CONFLICT (article_id, model_type) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    overall_sentiment = EXCLUDED.overall_sentiment,
                    overall_confidence = EXCLUDED.overall_confidence,
                    headline_sentiment = EXCLUDED.headline_sentiment,
                    headline_confidence = EXCLUDED.headline_confidence,
                    sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                    sentiment_aspects = EXCLUDED.sentiment_aspects,
                    processing_time_ms = EXCLUDED.processing_time_ms,
                    processed_at = NOW()
                """,
                [
                    (
                        a["article_id"],
                        a["model_type"],
                        a.get("model_name"),
                        a["overall_sentiment"],
                        a.get("overall_confidence"),
                        a["headline_sentiment"],
                        a.get("headline_confidence"),
                        a.get("sentiment_reasoning"),
                        json.dumps(a.get("sentiment_aspects")) if a.get("sentiment_aspects") else None,
                        a.get("processing_time_ms")
                    )
                    for a in analyses
                ],
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )

    def get_articles_without_sentiment(
        self,
        model_type: str,
        limit: int = None
    ) -> List[Dict]:
        """Get articles that haven't been analyzed for sentiment with given model.

        Args:
            model_type: Type of model ('llm', 'local', 'hybrid')
            limit: Maximum number of articles to return
        """
        schema = self.config["schema"]
        query = f"""
            SELECT a.id, a.title, a.content, a.date_posted, a.source_id
            FROM {schema}.news_articles a
            LEFT JOIN {schema}.sentiment_analyses sa
                ON a.id = sa.article_id AND sa.model_type = %s
            WHERE sa.id IS NULL
              AND a.content IS NOT NULL
              AND a.content != ''
              AND a.is_ditwah_cyclone = 1
            ORDER BY a.date_posted
        """
        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, (model_type,))
            return cur.fetchall()

    def get_sentiment_by_model(
        self,
        model_type: str = None,
        source_id: str = None,
        limit: int = None
    ) -> List[Dict]:
        """Get sentiment analysis results.

        Args:
            model_type: Filter by model type ('llm', 'local', 'hybrid')
            source_id: Filter by news source
            limit: Maximum number of results
        """
        schema = self.config["schema"]
        query = f"""
            SELECT sa.*, a.title, a.source_id, a.date_posted
            FROM {schema}.sentiment_analyses sa
            JOIN {schema}.news_articles a ON sa.article_id = a.id
            WHERE a.is_ditwah_cyclone = 1
        """
        params = []

        if model_type:
            query += " AND sa.model_type = %s"
            params.append(model_type)

        if source_id:
            query += " AND a.source_id = %s"
            params.append(source_id)

        query += " ORDER BY a.date_posted DESC"

        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_sentiment_comparison(self, limit: int = 100) -> List[Dict]:
        """Get sentiment results from all models for comparison.

        Args:
            limit: Maximum number of articles to compare
        """
        schema = self.config["schema"]
        query = f"""
            SELECT
                a.id as article_id,
                a.title,
                a.source_id,
                a.date_posted,
                MAX(CASE WHEN sa.model_type = 'llm' THEN sa.overall_sentiment END) as llm_sentiment,
                MAX(CASE WHEN sa.model_type = 'local' THEN sa.overall_sentiment END) as local_sentiment,
                MAX(CASE WHEN sa.model_type = 'hybrid' THEN sa.overall_sentiment END) as hybrid_sentiment,
                MAX(CASE WHEN sa.model_type = 'llm' THEN sa.sentiment_reasoning END) as llm_reasoning
            FROM {schema}.news_articles a
            JOIN {schema}.sentiment_analyses sa ON a.id = sa.article_id
            WHERE a.is_ditwah_cyclone = 1
            GROUP BY a.id, a.title, a.source_id, a.date_posted
            HAVING COUNT(DISTINCT sa.model_type) >= 2
            ORDER BY a.date_posted DESC
            LIMIT {limit}
        """

        with self.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    def get_sentiment_stats(self, model_type: str) -> Dict:
        """Get statistics for sentiment analysis by model type.

        Args:
            model_type: Type of model ('llm', 'local', 'hybrid')
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) as total_analyzed,
                    AVG(overall_sentiment) as avg_sentiment,
                    STDDEV(overall_sentiment) as stddev_sentiment,
                    MIN(overall_sentiment) as min_sentiment,
                    MAX(overall_sentiment) as max_sentiment,
                    AVG(overall_confidence) as avg_confidence
                FROM {schema}.sentiment_analyses
                WHERE model_type = %s
            """, (model_type,))
            return cur.fetchone()

    def refresh_sentiment_summary(self):
        """Refresh the sentiment_summary materialized view."""
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"REFRESH MATERIALIZED VIEW {schema}.sentiment_summary")

    # Article insights operations
    def get_article_by_id(self, article_id: int) -> Dict:
        """Fetch article metadata by ID.

        Args:
            article_id: The article ID

        Returns:
            Article dict with metadata or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id, date_posted, url, lang, is_ditwah_cyclone
                FROM {schema}.news_articles
                WHERE id = %s
            """, (article_id,))
            return cur.fetchone()

    def get_sentiment_for_article(self, article_id: int, model_type: str = 'roberta') -> Dict:
        """Fetch sentiment analysis for article and model.

        Args:
            article_id: The article ID
            model_type: The sentiment model type (default: roberta)

        Returns:
            Sentiment dict or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT overall_sentiment, headline_sentiment, overall_confidence,
                       headline_confidence, sentiment_reasoning, model_type, model_name
                FROM {schema}.sentiment_analyses
                WHERE article_id = %s AND model_type = %s
            """, (article_id, model_type))
            return cur.fetchone()

    def get_topic_for_article(self, article_id: int, version_id: str) -> Dict:
        """Fetch topic assignment for article.

        Args:
            article_id: The article ID
            version_id: The topic version ID

        Returns:
            Topic dict with name and confidence or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT t.id as topic_id, t.topic_id as bertopic_id, t.name as topic_name,
                       aa.topic_confidence, aa.overall_tone, aa.headline_tone
                FROM {schema}.article_analysis aa
                JOIN {schema}.topics t ON aa.primary_topic_id = t.id
                WHERE aa.article_id = %s AND aa.result_version_id = %s
                  AND t.result_version_id = %s
            """, (article_id, version_id, version_id))
            return cur.fetchone()

    def get_summary_for_article(self, article_id: int, version_id: str) -> Dict:
        """Fetch summary for article and version.

        Args:
            article_id: The article ID
            version_id: The summarization version ID

        Returns:
            Summary dict or None if not found
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT summary_text, method, compression_ratio, word_count,
                       sentence_count, summary_length, processing_time_ms
                FROM {schema}.article_summaries
                WHERE article_id = %s AND result_version_id = %s
            """, (article_id, version_id))
            return cur.fetchone()

    def get_cluster_for_article(self, article_id: int, version_id: str) -> Dict:
        """Fetch event cluster for article.

        Args:
            article_id: The article ID
            version_id: The clustering version ID

        Returns:
            Cluster dict with details or None if not in any cluster
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT ec.id as cluster_id, ec.cluster_name, ac.similarity_score,
                       ec.sources_count, ec.article_count, ec.date_start, ec.date_end,
                       ARRAY_AGG(DISTINCT n.source_id) as other_sources
                FROM {schema}.article_clusters ac
                JOIN {schema}.event_clusters ec ON ac.cluster_id = ec.id
                LEFT JOIN {schema}.article_clusters ac2 ON ec.id = ac2.cluster_id AND ac2.article_id != %s
                LEFT JOIN {schema}.news_articles n ON ac2.article_id = n.id
                WHERE ac.article_id = %s AND ac.result_version_id = %s
                  AND ec.result_version_id = %s
                GROUP BY ec.id, ec.cluster_name, ac.similarity_score,
                         ec.sources_count, ec.article_count, ec.date_start, ec.date_end
            """, (article_id, article_id, version_id, version_id))
            return cur.fetchone()

    def search_articles(self, title_search: str, limit: int = 50) -> List[Dict]:
        """Search articles by title.

        Args:
            title_search: Search term for title (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of article dicts with id, title, source_id, date_posted
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, source_id, date_posted
                FROM {schema}.news_articles
                WHERE title ILIKE %s
                ORDER BY date_posted DESC
                LIMIT %s
            """, (f"%{title_search}%", limit))
            return cur.fetchall()


# Convenience function
def get_db() -> Database:
    """Get a database connection."""
    return Database()
