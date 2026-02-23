"""Article database operations."""

from dataclasses import dataclass
from typing import Any, List, Dict, Optional


@dataclass
class ArticleFilter:
    """A single filter condition for article queries.

    Args:
        column: Column name to filter on (must be in ALLOWED_COLUMNS)
        op: SQL operator ('=', '!=', '>', '>=', '<', '<=', 'ILIKE')
        value: Value to compare against
    """
    column: str
    op: str
    value: Any


ALLOWED_COLUMNS = {
    "is_ditwah_cyclone", "date_posted", "source_id", "lang", "id"
}
ALLOWED_OPS = {"=", "!=", ">", ">=", "<", "<=", "ILIKE"}


def date_range_filters() -> List[ArticleFilter]:
    """Standard date range filters for the study period."""
    return [
        ArticleFilter("date_posted", ">=", "2025-11-22"),
        ArticleFilter("date_posted", "<=", "2025-12-31"),
    ]


def ditwah_filters() -> List[ArticleFilter]:
    """Standard filters for Ditwah cyclone analysis."""
    return [ArticleFilter("is_ditwah_cyclone", "=", 1)] + date_range_filters()


class ArticleMixin:
    """Article-related database operations."""

    def _build_filters(
        self,
        filters: Optional[List[ArticleFilter]],
        table_alias: str = ""
    ) -> tuple:
        """Build WHERE clause fragments and params from ArticleFilter list.

        Args:
            filters: List of ArticleFilter conditions
            table_alias: Optional table alias prefix (e.g. "a")

        Returns:
            Tuple of (clauses list, params list)
        """
        if not filters:
            return [], []
        clauses = []
        params = []
        prefix = f"{table_alias}." if table_alias else ""
        for f in filters:
            if f.column not in ALLOWED_COLUMNS:
                raise ValueError(f"Column '{f.column}' not allowed in filters")
            if f.op not in ALLOWED_OPS:
                raise ValueError(f"Operator '{f.op}' not allowed")
            clauses.append(f"{prefix}{f.column} {f.op} %s")
            params.append(f.value)
        return clauses, params

    def get_articles(
        self,
        limit: int = None,
        offset: int = 0,
        source_id: str = None,
        filters: List[ArticleFilter] = None
    ) -> List[Dict]:
        """Fetch articles from news_articles table."""
        schema = self.config["schema"]
        base_conditions = ["content IS NOT NULL", "content != ''"]
        params = []

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        if source_id:
            base_conditions.append("source_id = %s")
            params.append(source_id)

        where = " AND ".join(base_conditions)
        query = f"""
            SELECT id, url, title, content, date_posted, source_id, lang
            FROM {schema}.news_articles
            WHERE {where}
            ORDER BY date_posted, id
        """

        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_article_count(self, filters: List[ArticleFilter] = None) -> int:
        """Get total article count."""
        schema = self.config["schema"]
        base_conditions = ["content IS NOT NULL", "content != ''"]
        params = []

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.news_articles
                WHERE {where}
            """, params)
            return cur.fetchone()["count"]

    def get_article_by_url(
        self,
        url: str,
        filters: List[ArticleFilter] = None
    ) -> Dict:
        """Fetch article by URL.

        Args:
            url: The article URL to search for
            filters: Optional list of ArticleFilter conditions

        Returns:
            Article dict with id, url, title, content, source_id, date_posted, or None if not found
        """
        schema = self.config["schema"]
        base_conditions = ["url = %s"]
        params = [url]

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, url, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE {where}
            """, params)
            return cur.fetchone()

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

    def get_article_counts_by_source(
        self, filters: List[ArticleFilter] = None
    ) -> List[Dict]:
        """Get article counts grouped by source.

        Args:
            filters: Optional list of ArticleFilter conditions

        Returns:
            List of dicts with source_id and count
        """
        schema = self.config["schema"]
        base_conditions = []
        params = []

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = (" WHERE " + " AND ".join(base_conditions)) if base_conditions else ""
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT source_id, COUNT(*) as count
                FROM {schema}.news_articles
                {where}
                GROUP BY source_id
                ORDER BY count DESC
            """, params)
            return cur.fetchall()

    def get_article_date_range(
        self, filters: List[ArticleFilter] = None
    ) -> Dict:
        """Get min and max article dates.

        Args:
            filters: Optional list of ArticleFilter conditions

        Returns:
            Dict with min_date and max_date
        """
        schema = self.config["schema"]
        base_conditions = []
        params = []

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = (" WHERE " + " AND ".join(base_conditions)) if base_conditions else ""
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT MIN(date_posted)::date as min_date,
                       MAX(date_posted)::date as max_date
                FROM {schema}.news_articles
                {where}
            """, params)
            return cur.fetchone()

    def get_article_counts_by_date(
        self, filters: List[ArticleFilter] = None
    ) -> List[Dict]:
        """Get daily article counts grouped by date and source.

        Args:
            filters: Optional list of ArticleFilter conditions

        Returns:
            List of dicts with date, source_id, and count
        """
        schema = self.config["schema"]
        base_conditions = ["date_posted IS NOT NULL"]
        params = []

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT date_posted::date as date, source_id, COUNT(*) as count
                FROM {schema}.news_articles
                WHERE {where}
                GROUP BY date_posted::date, source_id
                ORDER BY date
            """, params)
            return cur.fetchall()

    def get_article_lengths(
        self, filters: List[ArticleFilter] = None
    ) -> List[Dict]:
        """Get article content lengths by source.

        Args:
            filters: Optional list of ArticleFilter conditions

        Returns:
            List of dicts with source_id and article_length
        """
        schema = self.config["schema"]
        base_conditions = ["content IS NOT NULL"]
        params = []

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT source_id, LENGTH(content) as article_length
                FROM {schema}.news_articles
                WHERE {where}
            """, params)
            return cur.fetchall()

    def search_articles(
        self,
        title_search: str,
        limit: int = 50,
        filters: List[ArticleFilter] = None
    ) -> List[Dict]:
        """Search articles by title.

        Args:
            title_search: Search term for title (case-insensitive)
            limit: Maximum number of results
            filters: Optional list of ArticleFilter conditions

        Returns:
            List of article dicts with id, title, source_id, date_posted
        """
        schema = self.config["schema"]
        base_conditions = ["title ILIKE %s"]
        params = [f"%{title_search}%"]

        filter_clauses, filter_params = self._build_filters(filters)
        base_conditions.extend(filter_clauses)
        params.extend(filter_params)

        where = " AND ".join(base_conditions)
        params.append(limit)
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, source_id, date_posted
                FROM {schema}.news_articles
                WHERE {where}
                ORDER BY date_posted DESC
                LIMIT %s
            """, params)
            return cur.fetchall()
