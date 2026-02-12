#!/usr/bin/env python3
"""Generate embeddings for a specific model."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings import generate_embeddings, get_embedding_stats


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for a specific model")
    parser.add_argument(
        "--model",
        default="all-mpnet-base-v2",
        help="Embedding model name (default: all-mpnet-base-v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for embedding generation (default: 1000)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to process (default: all)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Embedding Generation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print()

    generate_embeddings(
        embedding_model=args.model,
        batch_size=args.batch_size,
        limit=args.limit
    )

    stats = get_embedding_stats(embedding_model=args.model)
    print(f"\nEmbedding stats for model '{args.model}':")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  With embeddings: {stats['with_embeddings']}")
    print(f"  Completion: {stats['completion_pct']}%")


if __name__ == "__main__":
    main()
