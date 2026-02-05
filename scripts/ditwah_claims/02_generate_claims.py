#!/usr/bin/env python3
"""
Generate Ditwah Claims

Automatically generates claims from Ditwah articles and analyzes sentiment and stance.

Usage:
    python3 scripts/ditwah_claims/02_generate_claims.py --version-id <uuid>

Example:
    python3 scripts/ditwah_claims/02_generate_claims.py --version-id 123e4567-e89b-12d3-a456-426614174000
"""

import argparse
import logging
import sys
from uuid import UUID

from src.versions import get_version, update_pipeline_status
from src.ditwah_claims import generate_claims_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate claims from Ditwah articles and analyze sentiment/stance"
    )
    parser.add_argument(
        "--version-id",
        required=True,
        help="Result version ID (UUID)"
    )
    args = parser.parse_args()

    # Validate version
    try:
        version_id = UUID(args.version_id)
    except ValueError:
        logger.error(f"Invalid UUID: {args.version_id}")
        sys.exit(1)

    version = get_version(str(version_id))
    if not version:
        logger.error(f"Version not found: {version_id}")
        sys.exit(1)

    if version['analysis_type'] != 'ditwah_claims':
        logger.error(f"Version has wrong analysis_type: {version['analysis_type']} (expected 'ditwah_claims')")
        sys.exit(1)

    logger.info(f"Version: {version['name']}")
    logger.info(f"Description: {version['description']}")
    logger.info(f"Analysis type: {version['analysis_type']}")

    # Get configuration
    config = version['configuration']

    logger.info(f"Target claims: {config.get('generation', {}).get('num_claims', 15)}")
    logger.info(f"LLM provider: {config.get('llm', {}).get('provider', 'mistral')}")
    logger.info(f"LLM model: {config.get('llm', {}).get('model', 'mistral-large-latest')}")

    # Run pipeline
    logger.info("Starting claims generation pipeline...")
    summary = generate_claims_pipeline(version_id, config)

    # Check for errors
    if 'error' in summary:
        logger.error(f"Pipeline failed: {summary['error']}")
        sys.exit(1)

    # Mark pipeline as complete
    update_pipeline_status(str(version_id), 'ditwah_claims', True)
    logger.info("✅ Pipeline status updated")

    # Print summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"Claims generated: {summary['claims_generated']}")
    logger.info(f"Articles analyzed: {summary['articles_analyzed']}")
    logger.info(f"Sentiment records: {summary['sentiment_records']} stored in database")
    logger.info(f"Stance records: {summary['stance_records']} stored in database")
    logger.info("=" * 60)
    logger.info("Data stored in database tables:")
    logger.info("  - ditwah_claims")
    logger.info("  - claim_sentiment")
    logger.info("  - claim_stance")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
