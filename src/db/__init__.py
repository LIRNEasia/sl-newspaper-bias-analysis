"""Database connection and operations for media bias analysis."""

from src.config import load_config
from src.db._connection import DatabaseConnection
from src.db._articles import ArticleMixin, ArticleFilter, ditwah_filters, date_range_filters
from src.db._embeddings import EmbeddingMixin
from src.db._topics import TopicMixin
from src.db._clustering import ClusteringMixin
from src.db._sentiment import SentimentMixin
from src.db._ner import NERMixin
from src.db._entity_stance import EntityStanceMixin
from src.db._word_frequency import WordFrequencyMixin
from src.db._summaries import SummaryMixin


class Database(
    ArticleMixin,
    EmbeddingMixin,
    TopicMixin,
    ClusteringMixin,
    SentimentMixin,
    NERMixin,
    EntityStanceMixin,
    WordFrequencyMixin,
    SummaryMixin,
    DatabaseConnection,
):
    """PostgreSQL database connection manager."""
    pass


def get_db() -> Database:
    """Get a database connection."""
    return Database()


__all__ = ["Database", "get_db", "load_config", "ArticleFilter", "ditwah_filters", "date_range_filters"]
