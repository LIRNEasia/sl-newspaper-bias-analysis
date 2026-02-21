#!/usr/bin/env python3
"""Analyze entity stance in articles using NLI-based detection."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.entity_stance import entity_stance_pipeline
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Analyze entity stance in articles")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the entity_stance result version"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="NLI inference batch size (default: 16)"
    )
    args = parser.parse_args()

    # Get version and validate
    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "entity_stance":
        print(f"Error: Version {args.version_id} is not an entity_stance version "
              f"(type: {version['analysis_type']})")
        sys.exit(1)

    # Get version configuration
    version_config = get_version_config(args.version_id)

    # Validate NER version
    ner_version_id = version_config.get("ner_version_id")
    if not ner_version_id:
        print("Error: ner_version_id must be set in the version configuration")
        sys.exit(1)

    ner_version = get_version(ner_version_id)
    if not ner_version:
        print(f"Error: NER version {ner_version_id} not found")
        sys.exit(1)

    if ner_version["analysis_type"] != "ner":
        print(f"Error: Version {ner_version_id} is not an NER version")
        sys.exit(1)

    if not ner_version.get("is_complete"):
        print(f"Error: NER version {ner_version_id} pipeline is not complete. "
              "Run NER extraction first.")
        sys.exit(1)

    print("=" * 60)
    print("Entity Stance Detection (NLI-based)")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print(f"NER Version: {ner_version['name']} ({ner_version_id})")
    print()

    stance_config = version_config.get("entity_stance", {})
    print(f"Model: {stance_config.get('model', 'cross-encoder/nli-deberta-v3-base')}")
    print(f"Chunk size: {stance_config.get('chunk_size', 5)} sentences")
    print(f"Neutral threshold: {stance_config.get('neutral_threshold', 0.2)}")
    print(f"Min confidence: {stance_config.get('min_confidence', 0.3)}")
    print(f"Entity types: {stance_config.get('entity_types', [])}")
    if args.limit:
        print(f"Article limit: {args.limit}")
    print()

    # Run pipeline
    summary = entity_stance_pipeline(
        version_id=args.version_id,
        config=version_config,
        limit=args.limit,
        batch_size=args.batch_size
    )

    # Update pipeline status
    update_pipeline_status(args.version_id, "entity_stance", True)
    print(f"\n✓ Entity stance step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
