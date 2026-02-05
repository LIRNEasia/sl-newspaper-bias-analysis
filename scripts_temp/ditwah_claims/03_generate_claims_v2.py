#!/usr/bin/env python3
"""
Generate Ditwah Claims - Version 2

New approach:
- Generate claims FROM each individual article
- Deduplicate similar claims
- Ensure all claims have >=5 articles
- Ensure all claims have both sentiment AND stance data

Usage:
    python3 scripts/ditwah_claims/03_generate_claims_v2.py --version-id <uuid>

Example:
    python3 scripts/ditwah_claims/03_generate_claims_v2.py --version-id 123e4567-e89b-12d3-a456-426614174000
"""

import argparse
import logging
import sys
from uuid import UUID

from src.versions import get_version, update_pipeline_status
from src.ditwah_claims_v2 import generate_claims_pipeline_v2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate claims from Ditwah articles (per-article approach)"
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

    # Display version info
    logger.info(f"Version: {version['name']}")
    logger.info(f"Description: {version['description']}")
    logger.info(f"Analysis type: {version['analysis_type']}")

    config = version['configuration']
    generation_config = config.get('generation', {})
    llm_config = config.get('llm', {})

    logger.info(f"Min articles per claim: {generation_config.get('min_articles', 5)}")
    logger.info(f"LLM provider: {llm_config.get('provider', 'mistral')}")
    logger.info(f"LLM model: {llm_config.get('model', 'mistral-large-latest')}")

    logger.info("\nStarting claims generation pipeline V2...")
    logger.info("This will:")
    logger.info("  1. Generate 1-3 claims from EACH Ditwah article")
    logger.info("  2. Deduplicate similar claims using LLM")
    logger.info("  3. Filter claims with < 5 articles")
    logger.info("  4. Analyze sentiment and stance for all claims")
    logger.info("  5. Keep only claims with BOTH sentiment and stance data\n")

    # Run pipeline
    try:
        summary = generate_claims_pipeline_v2(version_id, config)

        if 'error' in summary:
            logger.error(f"Pipeline failed: {summary['error']}")
            sys.exit(1)

        # Display summary
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Ditwah articles: {summary['total_articles']}")
        logger.info(f"Articles that generated claims: {summary['articles_with_claims']}")
        logger.info(f"Raw claims generated: {summary['raw_claims']}")
        logger.info(f"After deduplication: {summary['deduplicated_claims']}")
        logger.info(f"After filtering (≥5 articles): {summary['filtered_claims']}")
        logger.info(f"Claims with sentiment & stance: {summary['claims_with_data']}")
        logger.info(f"Total sentiment records: {summary['sentiment_records']}")
        logger.info(f"Total stance records: {summary['stance_records']}")
        logger.info("="*60)

        # Update pipeline status
        update_pipeline_status(str(version_id), 'ditwah_claims', True)
        logger.info(f"\n✅ Pipeline complete! Version {version_id} is ready.")

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
