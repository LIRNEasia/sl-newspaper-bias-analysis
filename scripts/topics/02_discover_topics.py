#!/usr/bin/env python3
"""Discover topics using BERTopic for topic analysis."""

import os
import sys
import argparse
from pathlib import Path


# Set environment variables for single-threaded execution (reproducibility)
# These prevent NumPy, BLAS, OpenMP from using parallelism
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.topics import discover_topics
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Discover topics using BERTopic")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the topic result version"
    )
    parser.add_argument(
        "--nr-topics",
        type=int,
        default=None,
        help="Target number of topics (default: auto)"
    )
    args = parser.parse_args()

    # Get version and validate it's a topic version
    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "topics":
        print(f"Error: Version {args.version_id} is not a topic analysis version (type: {version['analysis_type']})")
        print("Use scripts/topics/ for topic analysis versions only")
        sys.exit(1)

    # Get version configuration
    version_config = get_version_config(args.version_id)

    print("=" * 60)
    print("Topic Discovery with BERTopic")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print()

    # Extract topic and embeddings configuration
    topic_config = version_config.get("topics", {})
    embeddings_config = version_config.get("embeddings", {})

    # Discover topics
    summary = discover_topics(
        result_version_id=args.version_id,
        topic_config=topic_config,
        embeddings_config=embeddings_config,
        nr_topics=args.nr_topics
    )

    # Print discovered topics
    print("\n" + "=" * 60)
    print("Discovered Topics:")
    print("=" * 60)

    for topic in sorted(summary["topics"], key=lambda x: x["article_count"], reverse=True):
        if topic["topic_id"] == -1:
            continue
        print(f"\n[Topic {topic['topic_id']}] {topic['name']}")
        print(f"  Articles: {topic['article_count']}")
        print(f"  Keywords: {', '.join(topic['keywords'][:5])}")
        if topic["description"]:
            print(f"  Description: {topic['description']}")

    # Update pipeline status
    update_pipeline_status(args.version_id, "topics", True)
    print(f"\n✓ Topics step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
