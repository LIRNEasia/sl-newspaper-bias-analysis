#!/usr/bin/env python3
"""Cluster articles into events for clustering analysis."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clustering import cluster_articles
from src.versions import get_version, get_version_config, update_pipeline_status


def main():
    parser = argparse.ArgumentParser(description="Cluster articles into events")
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the clustering result version"
    )
    args = parser.parse_args()

    # Get version and validate it's a clustering version
    version = get_version(args.version_id)
    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "clustering":
        print(f"Error: Version {args.version_id} is not a clustering analysis version (type: {version['analysis_type']})")
        print("Use scripts/clustering/ for clustering analysis versions only")
        sys.exit(1)

    # Get version configuration
    version_config = get_version_config(args.version_id)

    print("=" * 60)
    print("Event Clustering")
    print("=" * 60)
    print(f"Version: {version['name']}")
    print(f"Version ID: {args.version_id}")
    print()

    # Extract clustering configuration
    cluster_config = version_config.get("clustering", {})

    summary = cluster_articles(
        result_version_id=args.version_id,
        similarity_threshold=cluster_config.get("similarity_threshold", 0.8),
        time_window_days=cluster_config.get("time_window_days", 7),
        min_cluster_size=cluster_config.get("min_cluster_size", 2)
    )

    print("\n" + "=" * 60)
    print("Clustering Summary:")
    print("=" * 60)
    print(f"Total clusters: {summary['total_clusters']}")
    print(f"Articles clustered: {summary['articles_clustered']}")
    print(f"Multi-source clusters: {summary['multi_source_clusters']}")

    # Update pipeline status
    update_pipeline_status(args.version_id, "clustering", True)
    print(f"\n✓ Clustering step marked complete for version {args.version_id}")


if __name__ == "__main__":
    main()
