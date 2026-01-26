"""Database connection and operations for media bias analysis."""

import yaml
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


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
            WHERE content IS NOT NULL AND content != ''
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
                WHERE content IS NOT NULL AND content != ''
            """)
            return cur.fetchone()["count"]

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
                ORDER BY a.date_posted, a.id
            """
            params = [result_version_id]
        else:
            query = f"""
                SELECT e.article_id, e.embedding::text, a.title, a.content,
                       a.date_posted, a.source_id
                FROM {schema}.embeddings e
                JOIN {schema}.news_articles a ON e.article_id = a.id
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


# Convenience function
def get_db() -> Database:
    """Get a database connection."""
    return Database()
