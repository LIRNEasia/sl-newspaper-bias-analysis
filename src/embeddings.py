"""Embedding generation for articles."""

import random
import numpy as np
from typing import Dict
from tqdm import tqdm

from .db import get_db
from .llm import get_embeddings_client, load_config


def generate_embeddings(
    result_version_id: str,
    batch_size: int = 50,
    limit: int = None,
    show_progress: bool = True,
    random_seed: int = 42
):
    """
    Generate embeddings for all articles that don't have them yet for a specific version.

    Args:
        result_version_id: UUID of the result version
        batch_size: Number of articles to process per batch
        limit: Maximum articles to process (None = all)
        show_progress: Show progress bar
        random_seed: Random seed for reproducibility
    """
    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    config = load_config()
    embed_client = get_embeddings_client(config.get("embeddings"))

    with get_db() as db:
        # Get articles without embeddings for this version
        articles = db.get_articles_without_embeddings(result_version_id=result_version_id, limit=limit)
        total = len(articles)

        if total == 0:
            print(f"All articles already have embeddings for version {result_version_id}.")
            return

        print(f"Generating embeddings for {total} articles (version: {result_version_id})...")

        # Process in batches
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

            db.store_embeddings(embedding_records, result_version_id)
            processed += len(batch)

        print(f"Generated embeddings for {processed} articles.")


def get_embedding_stats() -> Dict:
    """Get statistics about embeddings."""
    with get_db() as db:
        total_articles = db.get_article_count()
        embedded = db.get_embedding_count()

        return {
            "total_articles": total_articles,
            "with_embeddings": embedded,
            "without_embeddings": total_articles - embedded,
            "completion_pct": round(100 * embedded / total_articles, 1) if total_articles > 0 else 0
        }


if __name__ == "__main__":
    print("Please use scripts/01_generate_embeddings.py instead.")
    print("Usage: python3 scripts/01_generate_embeddings.py --version-id <uuid>")
