"""Embedding database operations."""

from typing import List, Dict, Any
from psycopg2.extras import execute_values


class EmbeddingMixin:
    """Embedding-related database operations."""

    def store_embeddings(self, embeddings: List[Dict[str, Any]]):
        """Store article embeddings in batch.

        Args:
            embeddings: List of dicts with 'article_id', 'embedding', and 'model' keys
        """
        schema = self.config["schema"]
        with self.cursor(dict_cursor=False) as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {schema}.embeddings (article_id, embedding, embedding_model)
                VALUES %s
                ON CONFLICT (article_id, embedding_model) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
                """,
                [
                    (e["article_id"], e["embedding"], e.get("model", "all-mpnet-base-v2"))
                    for e in embeddings
                ],
                template="(%s, %s::vector, %s)"
            )

    def get_articles_without_embeddings(
        self,
        embedding_model: str,
        limit: int = None,
        filters: list = None
    ) -> List[Dict]:
        """Get articles that don't have embeddings yet for a specific model.

        Args:
            embedding_model: Name of the embedding model (e.g., 'all-mpnet-base-v2')
            limit: Maximum number of articles to return
            filters: Optional list of ArticleFilter conditions (applied to article table)
        """
        schema = self.config["schema"]

        base_conditions = [
            "e.id IS NULL",
            "a.content IS NOT NULL",
            "a.content != ''"
        ]
        params = [embedding_model]

        filter_clauses, filter_params = self._build_filters(filters, table_alias="a")
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        query = f"""
            SELECT a.id, a.title, a.content, a.date_posted, a.source_id
            FROM {schema}.news_articles a
            LEFT JOIN {schema}.embeddings e ON a.id = e.article_id AND e.embedding_model = %s
            WHERE {where}
            ORDER BY a.date_posted, a.id
        """

        if limit:
            query += f" LIMIT {limit}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_all_embeddings(
        self,
        embedding_model: str,
        filters: list = None
    ) -> List[Dict]:
        """Get all article embeddings for a specific model.

        Args:
            embedding_model: Name of the embedding model (e.g., 'all-mpnet-base-v2')
            filters: Optional list of ArticleFilter conditions (applied to article table)
        """
        schema = self.config["schema"]

        base_conditions = ["e.embedding_model = %s"]
        params = [embedding_model]

        filter_clauses, filter_params = self._build_filters(filters, table_alias="a")
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        query = f"""
            SELECT e.article_id, e.embedding::text, a.title, a.content,
                   a.date_posted, a.source_id
            FROM {schema}.embeddings e
            JOIN {schema}.news_articles a ON e.article_id = a.id
            WHERE {where}
            ORDER BY a.date_posted, a.id
        """

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

    def get_embedding_count(self, embedding_model: str = None) -> int:
        """Get count of articles with embeddings, optionally filtered by model.

        Args:
            embedding_model: Optional model name filter. If None, counts all embeddings.
        """
        schema = self.config["schema"]
        with self.cursor() as cur:
            if embedding_model:
                cur.execute(
                    f"SELECT COUNT(*) as count FROM {schema}.embeddings WHERE embedding_model = %s",
                    (embedding_model,)
                )
            else:
                cur.execute(f"SELECT COUNT(*) as count FROM {schema}.embeddings")
            return cur.fetchone()["count"]
