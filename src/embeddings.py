"""Embedding generation for articles."""

import random
import numpy as np
from typing import Dict
from tqdm import tqdm

from .db import get_db, ditwah_filters
from .llm import get_embeddings_client, load_config


def generate_embeddings(
    embedding_model: str = "all-mpnet-base-v2",
    batch_size: int = 50,
    limit: int = None,
    show_progress: bool = True,
    random_seed: int = 42,
    embeddings_config: dict = None
):
    """
    Generate embeddings for all articles that don't have them yet for a specific model.

    Args:
        embedding_model: Name of the embedding model (e.g., 'all-mpnet-base-v2')
        batch_size: Number of articles to process per batch
        limit: Maximum articles to process (None = all)
        show_progress: Show progress bar
        random_seed: Random seed for reproducibility
        embeddings_config: Optional override for provider, matryoshka, etc.
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Build config: start from global defaults, override model name
    config = load_config()
    config_dict = config.get("embeddings", {})
    if embeddings_config:
        config_dict.update(embeddings_config)
    config_dict["model"] = embedding_model

    # For EmbeddingGemma, always use "clustering" task for consistency (shared embeddings)
    embed_client = get_embeddings_client(config_dict)

    with get_db() as db:
        articles = db.get_articles_without_embeddings(
            embedding_model=embedding_model, limit=limit, filters=ditwah_filters()
        )
        total = len(articles)

        if total == 0:
            print(f"All articles already have embeddings for model '{embedding_model}'.")
            return

        print(f"Generating embeddings for {total} articles (model: {embedding_model})...")

        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches")

        processed = 0
        for i in iterator:
            batch = articles[i:i + batch_size]

            # Prepare texts: title + content
            texts = [
                f"{a['title']}\n\n{a['content'][:8000]}"  # Truncate long content
                for a in batch
            ]

            embeddings = embed_client.embed(texts)

            # Prepare for storage
            embedding_records = [
                {
                    "article_id": str(batch[j]["id"]),
                    "embedding": embeddings[j],
                    "model": embed_client.model
                }
                for j in range(len(batch))
            ]

            db.store_embeddings(embedding_records)
            processed += len(batch)

        print(f"Generated embeddings for {processed} articles.")


def get_embedding_stats(embedding_model: str = None) -> Dict:
    """Get statistics about embeddings.

    Args:
        embedding_model: Optional model name filter. If None, counts all embeddings.
    """
    with get_db() as db:
        total_articles = db.get_article_count(filters=ditwah_filters())
        embedded = db.get_embedding_count(embedding_model=embedding_model)

        return {
            "total_articles": total_articles,
            "with_embeddings": embedded,
            "without_embeddings": total_articles - embedded,
            "completion_pct": round(100 * embedded / total_articles, 1) if total_articles > 0 else 0
        }


if __name__ == "__main__":
    print("Please use scripts/embeddings/01_generate_embeddings.py instead.")
    print("Usage: python3 scripts/embeddings/01_generate_embeddings.py --model <model-name>")
