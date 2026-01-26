#!/usr/bin/env python3
"""Generate embeddings for topic analysis."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings import generate_embeddings
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for topic analysis")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the topic result version"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for embedding generation (default: 1000)"
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
    embeddings_config = version_config.get("embeddings", {})
    random_seed = version_config.get("random_seed", 42)

    print(f"\nGenerating embeddings for topic analysis version: {version['name']}")
    print(f"  Version ID: {args.version_id}")
    print(f"  Model: {embeddings_config.get('model', 'all-mpnet-base-v2')}")
    print(f"  Random seed: {random_seed}")
    print(f"  Batch size: {args.batch_size}\n")

    # Generate embeddings
    generate_embeddings(
        result_version_id=args.version_id,
        batch_size=args.batch_size,
        random_seed=random_seed
    )

    # Update pipeline status
    update_pipeline_status(args.version_id, "embeddings", True)
    print(f"\n✓ Embeddings step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
