"""Topic modeling using BERTopic for hierarchical topic discovery."""

import random
import numpy as np
from typing import List, Dict, Tuple

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

from .db import get_db, load_config


class TopicModeler:
    """Discovers topics from article corpus using BERTopic."""

    def __init__(
        self,
        min_topic_size: int = 10,
        diversity: float = 0.5,
        nr_topics: int = None,
        embedding_model: str = "all-mpnet-base-v2",
        stop_words: List[str] = None,
        random_seed: int = 42,
        umap_params: Dict = None,
        hdbscan_params: Dict = None,
        vectorizer_params: Dict = None
    ):
        # Set random seeds for reproducibility
        # Note: PYTHONHASHSEED must be set before Python starts (see script header)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load embedding model for word representations
        print(f"Loading embedding model for topic representation...")
        self.embedding_model = SentenceTransformer(embedding_model)

        # UMAP parameters with defaults
        umap_defaults = {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": random_seed
        }
        if umap_params:
            umap_defaults.update(umap_params)

        # UMAP for dimensionality reduction
        self.umap_model = UMAP(**umap_defaults)

        # HDBSCAN parameters with defaults
        hdbscan_defaults = {
            "min_cluster_size": min_topic_size,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
            "core_dist_n_jobs": 1  # Force single-threaded for reproducibility
        }
        if hdbscan_params:
            hdbscan_defaults.update(hdbscan_params)

        # HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(**hdbscan_defaults)

        # Vectorizer for topic representation
        # Use provided stop_words or default to domain stop words
        if stop_words is None:
            stop_words = []

        custom_stop_words = sorted(set(ENGLISH_STOP_WORDS) | set(stop_words))

        vectorizer_defaults = {
            "ngram_range": (1, 3),
            "stop_words": custom_stop_words,
            "min_df": 5
        }
        if vectorizer_params:
            # Don't override stop_words from vectorizer_params
            vectorizer_params_copy = vectorizer_params.copy()
            if "ngram_range" in vectorizer_params_copy:
                vectorizer_params_copy["ngram_range"] = tuple(vectorizer_params_copy["ngram_range"])
            vectorizer_defaults.update(vectorizer_params_copy)

        self.vectorizer = CountVectorizer(**vectorizer_defaults)

        # Representation model
        self.representation_model = KeyBERTInspired()

        # Initialize BERTopic with embedding model
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer,
            representation_model=self.representation_model,
            nr_topics=nr_topics,
            verbose=True
        )

        self.topics = None
        self.probs = None
        self.topic_info = None

    def fit(
        self,
        documents: List[str],
        embeddings: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Fit BERTopic model using pre-computed embeddings.

        Args:
            documents: List of article texts
            embeddings: Pre-computed embeddings array

        Returns:
            topics: Topic assignment for each document
            probs: Topic probabilities
        """
        print(f"Fitting BERTopic on {len(documents)} documents...")
        self.topics, self.probs = self.model.fit_transform(documents, embeddings)
        self.topic_info = self.model.get_topic_info()

        n_topics = len(self.topic_info) - 1  # Exclude -1 (outliers)
        n_outliers = sum(1 for t in self.topics if t == -1)
        print(f"Discovered {n_topics} topics ({n_outliers} outliers)")

        return self.topics, self.probs

    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[str]:
        """Get top keywords for a topic."""
        topic_words = self.model.get_topic(topic_id)
        if topic_words:
            return [word for word, _ in topic_words[:n_words]]
        return []

    def get_representative_docs(self, topic_id: int, n_docs: int = 3) -> List[str]:
        """Get representative documents for a topic."""
        docs = self.model.get_representative_docs(topic_id)
        return docs[:n_docs] if docs else []

    def reduce_topics(self, documents: List[str], nr_topics: int):
        """Reduce number of topics."""
        print(f"Reducing to {nr_topics} topics...")
        self.topics = self.model.reduce_topics(documents, nr_topics=nr_topics)
        self.topic_info = self.model.get_topic_info()
        return self.topics

    def save(self, path: str):
        """Save the model."""
        self.model.save(path, serialization="safetensors", save_ctfidf=True)

    @classmethod
    def load(cls, path: str) -> "TopicModeler":
        """Load a saved model."""
        instance = cls.__new__(cls)
        instance.model = BERTopic.load(path)
        instance.topic_info = instance.model.get_topic_info()
        return instance


def label_topics_from_keywords(topic_modeler: TopicModeler) -> List[Dict]:
    """Generate topic labels from keywords (no LLM needed)."""
    labeled_topics = []
    topic_info = topic_modeler.topic_info

    print("Generating topic labels from keywords...")

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]

        if topic_id == -1:  # Outliers
            labeled_topics.append({
                "topic_id": -1,
                "name": "Uncategorized",
                "description": "Articles that don't fit into any specific topic",
                "keywords": [],
                "article_count": row["Count"]
            })
            continue

        keywords = topic_modeler.get_topic_keywords(topic_id)

        # Create name from top 3 keywords
        name = ", ".join(keywords[:3]).title()

        labeled_topics.append({
            "topic_id": topic_id,
            "name": name,
            "description": f"Articles about: {', '.join(keywords[:5])}",
            "keywords": keywords,
            "article_count": row["Count"]
        })

    return labeled_topics


def discover_topics(
    result_version_id: str,
    topic_config: Dict = None,
    nr_topics: int = None,
    save_model: bool = True
) -> Dict:
    """
    Main function to discover topics from the article corpus for a specific version.

    Args:
        result_version_id: UUID of the result version
        topic_config: Topic configuration (from version config, or uses defaults from config.yaml)
        nr_topics: Target number of topics (None = auto)
        save_model: Whether to save the trained model

    Returns:
        Summary of discovered topics
    """
    # Load default config if not provided
    if topic_config is None:
        config = load_config()
        topic_config = config.get("topics", {})

    # Load articles with embeddings for this version
    print(f"Loading articles and embeddings for version {result_version_id}...")
    with get_db() as db:
        data = db.get_all_embeddings(result_version_id=result_version_id)

    print(f"Loaded {len(data)} articles with embeddings")

    # Prepare data
    documents = [f"{d['title']}\n\n{d['content'][:8000]}" for d in data]
    embeddings = np.array([d['embedding'] for d in data])
    article_ids = [str(d['article_id']) for d in data]

    # Create and fit topic model with custom parameters
    modeler = TopicModeler(
        min_topic_size=topic_config.get("min_topic_size", 10),
        diversity=topic_config.get("diversity", 0.5),
        nr_topics=nr_topics or topic_config.get("nr_topics"),
        embedding_model=topic_config.get("embedding_model", "all-mpnet-base-v2"),
        stop_words=topic_config.get("stop_words"),
        random_seed=topic_config.get("random_seed", 42),
        umap_params=topic_config.get("umap"),
        hdbscan_params=topic_config.get("hdbscan"),
        vectorizer_params=topic_config.get("vectorizer")
    )

    topics, probs = modeler.fit(documents, embeddings)

    # Label topics from keywords (no LLM needed)
    labeled_topics = label_topics_from_keywords(modeler)

    # Save to database
    print("Saving topics to database...")
    with get_db() as db:
        # Store topics
        db.store_topics(labeled_topics, result_version_id)

        # Store article-topic assignments
        assignments = [
            {
                "article_id": article_ids[i],
                "topic_id": topics[i],
                "confidence": float(probs[i]) if probs is not None else 0.0
            }
            for i in range(len(article_ids))
            if topics[i] != -1  # Skip outliers for now
        ]
        db.store_article_topics(assignments, result_version_id)

    # Save model
    if save_model:
        model_path = f"models/bertopic_model_{result_version_id[:8]}"
        import os
        os.makedirs("models", exist_ok=True)
        modeler.save(model_path)
        print(f"Model saved to {model_path}")

    # Summary
    n_topics = len([t for t in labeled_topics if t["topic_id"] != -1])
    n_outliers = sum(1 for t in topics if t == -1)

    summary = {
        "total_articles": len(documents),
        "topics_discovered": n_topics,
        "outliers": n_outliers,
        "topics": labeled_topics
    }

    print(f"\nTopic Discovery Complete:")
    print(f"  Total articles: {len(documents)}")
    print(f"  Topics discovered: {n_topics}")
    print(f"  Outliers: {n_outliers}")

    return summary


if __name__ == "__main__":
    print("Please use scripts/topics/02_discover_topics.py instead.")
    print("Usage: python3 scripts/topics/02_discover_topics.py --version-id <uuid>")
