#!/usr/bin/env python3
"""Compute word frequencies for word frequency analysis."""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.word_frequency import compute_word_frequencies
from src.versions import get_version, get_version_config


def main():
    parser = argparse.ArgumentParser(
        description="Compute word frequencies for a word_frequency result version"
    )
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the word_frequency result version"
    )
    args = parser.parse_args()

    # Get version and validate
    print(f"Loading version {args.version_id}...")
    version = get_version(args.version_id)

    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "word_frequency":
        print(f"Error: Version '{version['name']}' is not a word_frequency analysis version")
        print(f"       Found analysis_type: {version['analysis_type']}")
        sys.exit(1)

    print(f"Version: {version['name']}")
    print(f"Description: {version.get('description', 'N/A')}")
    print(f"Analysis type: {version['analysis_type']}")

    # Get configuration
    version_config = get_version_config(args.version_id)
    wf_config = version_config.get("word_frequency", {})

    if not wf_config:
        print("Error: No word_frequency configuration found in version")
        sys.exit(1)

    print(f"\nWord Frequency Configuration:")
    print(f"  Ranking method: {wf_config.get('ranking_method', 'frequency')}")
    print(f"  TF-IDF scope: {wf_config.get('tfidf_scope', 'per_source')}")
    print(f"  Top N words: {wf_config.get('top_n_words', 50)}")
    print(f"  Min word length: {wf_config.get('min_word_length', 3)}")
    print(f"  Custom stopwords: {len(wf_config.get('custom_stopwords', []))} words")

    # Compute word frequencies
    print("\n" + "=" * 60)
    summary = compute_word_frequencies(args.version_id, wf_config)
    print("=" * 60)

    print("\nWord frequency analysis completed successfully!")
    print(f"Pipeline status updated for version: {args.version_id}")


if __name__ == "__main__":
    main()
