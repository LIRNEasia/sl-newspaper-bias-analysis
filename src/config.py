"""Centralized configuration loading for media bias analysis."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml.

    Returns:
        Dictionary containing all configuration sections.

    Raises:
        FileNotFoundError: If config.yaml doesn't exist
        yaml.YAMLError: If config.yaml is invalid
    """
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please create config.yaml from config.yaml.example"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_database_config() -> Dict[str, Any]:
    """Get database configuration section."""
    return load_config()["database"]


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration section."""
    return load_config()["llm"]


def get_embeddings_config() -> Dict[str, Any]:
    """Get embeddings configuration section."""
    return load_config()["embeddings"]


def get_topics_config() -> Dict[str, Any]:
    """Get topics configuration section."""
    return load_config()["topics"]


def get_clustering_config() -> Dict[str, Any]:
    """Get clustering configuration section."""
    return load_config()["clustering"]


def get_summarization_config() -> Dict[str, Any]:
    """Get summarization configuration section."""
    return load_config()["summarization"]


def get_sentiment_config() -> Dict[str, Any]:
    """Get sentiment analysis configuration section."""
    return load_config()["sentiment"]
