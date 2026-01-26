#!/usr/bin/env python3
"""Extract named entities from articles."""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ner import extract_entities_from_articles
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Extract named entities from articles")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the NER result version"
    )
    args = parser.parse_args()

    # Get version and validate it's an NER version
    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "ner":
        print(f"Error: Version {args.version_id} is not an NER analysis version (type: {version['analysis_type']})")
        print("Use scripts/ner/ for NER analysis versions only")
        sys.exit(1)

    # Get version configuration
    version_config = get_version_config(args.version_id)

    print("=" * 60)
    print("Named Entity Recognition (NER) Extraction")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print()

    # Extract NER configuration
    ner_config = version_config.get("ner", {})

    # Extract entities
    summary = extract_entities_from_articles(
        result_version_id=args.version_id,
        ner_config=ner_config,
        batch_size=ner_config.get("batch_size", 32)
    )

    # Update pipeline status
    update_pipeline_status(args.version_id, "ner", True)
    print(f"\n✓ NER step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
