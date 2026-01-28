#!/usr/bin/env python3
"""Generate article summaries for summarization analysis."""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.versions import get_version, get_version_config, update_pipeline_status
from src.summarization import generate_summaries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate article summaries for a summarization version"
    )
    parser.add_argument(
        "--version-id",
        required=True,
        help="UUID of the summarization version"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of articles to process in each batch (default: 50)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of articles to summarize. Defaults to 10 for LLM methods (claude/gpt), unlimited for others."
    )
    args = parser.parse_args()

    # Validate version
    print(f"Loading version {args.version_id}...")
    version = get_version(args.version_id)

    if not version:
        print(f"Error: Version {args.version_id} not found")
        sys.exit(1)

    if version["analysis_type"] != "summarization":
        print(f"Error: Version {args.version_id} is not a summarization version")
        print(f"  Analysis type: {version['analysis_type']}")
        sys.exit(1)

    # Get configuration
    config = get_version_config(args.version_id)
    summ_config = config.get("summarization", {})

    print(f"Version: {version['name']}")
    print(f"Description: {version['description']}")
    method = summ_config.get('method', 'textrank')
    print(f"Method: {method}")
    print(f"Summary length: {summ_config.get('summary_length', 'medium')}")

    # Default limit to 10 for LLM methods to control costs
    limit = args.limit
    if limit is None and method in ("claude", "gpt"):
        limit = 10
        print(f"Note: Limiting to {limit} articles for LLM method (use --limit to override)")
    if limit:
        print(f"Article limit: {limit}")
    print()

    # Generate summaries
    try:
        summary = generate_summaries(
            result_version_id=args.version_id,
            summarization_config=summ_config,
            batch_size=args.batch_size,
            limit=limit
        )

        # Update pipeline status
        print("\nUpdating pipeline status...")
        update_pipeline_status(args.version_id, "summarization", True)

        print(f"\n✓ Pipeline complete for version {args.version_id}")
        print(f"  Generated {summary['successful']} summaries")
        print(f"  Average compression: {summary['avg_compression']:.1f}%")
        print(f"  Average processing time: {summary['avg_time_ms']:.0f}ms")

    except Exception as e:
        print(f"\n✗ Error generating summaries: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
