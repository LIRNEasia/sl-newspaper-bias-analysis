#!/usr/bin/env python3
"""
Analyze Sentiment and Stance for General Claims

This script:
1. Links existing sentiment data to general claims
2. Generates stance analysis (agree/disagree/neutral) for each article-claim pair

This is step 3 of the two-step claims generation process.

Usage:
    python3 scripts/ditwah_claims/04_analyze_sentiment_stance.py --version-id <uuid>

Prerequisites:
    1. Run 02_generate_individual_claims.py
    2. Run 03_cluster_claims.py
    3. Ensure sentiment analysis has been run on DITWAH articles

Output:
    - claim_sentiment table populated
    - claim_stance table populated
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_db
from src.llm import get_llm
from src.ditwah_claims import (
    get_articles_for_general_claim,
    link_sentiment_to_general_claims,
    analyze_claim_stance
)
from src.versions import get_version

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment and stance for general claims')
    parser.add_argument('--version-id', type=str, required=True,
                       help='Result version ID (UUID)')
    parser.add_argument('--sentiment-model', type=str, default='roberta',
                       help='Sentiment model to use (default: roberta)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("DITWAH Sentiment & Stance Analysis Pipeline")
    logger.info("=" * 80)

    # Get version and config
    version = get_version(args.version_id)
    if not version:
        logger.error(f"Version not found: {args.version_id}")
        sys.exit(1)

    logger.info(f"Version: {version['name']}")
    logger.info(f"Description: {version['description']}")

    config = version['configuration']
    llm_config = config.get('llm', {})
    stance_config = config.get('stance', {})

    # Check if general claims exist
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.ditwah_claims
                WHERE result_version_id = %s
            """, (args.version_id,))
            general_count = cur.fetchone()['count']

    if general_count == 0:
        logger.error("❌ No general claims found. Run 03_cluster_claims.py first.")
        sys.exit(1)

    logger.info(f"✅ Found {general_count} general claims")

    # Link sentiment data
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Linking Sentiment Data to General Claims")
    logger.info("=" * 80)
    logger.info(f"Using sentiment model: {args.sentiment_model}")

    sentiment_count = link_sentiment_to_general_claims(
        version_id=args.version_id,
        sentiment_model=args.sentiment_model
    )

    logger.info(f"✅ Linked {sentiment_count} sentiment records")

    # Analyze stance for each general claim
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Analyzing Stance for General Claims")
    logger.info("=" * 80)
    logger.info(f"Provider: {llm_config.get('provider', 'local')}")
    logger.info(f"Model: {llm_config.get('model', 'llama3.1:latest')}")
    logger.info("This may take 1-2 hours with local LLM")
    logger.info("")

    # Initialize LLM
    llm = get_llm(llm_config)
    llm_provider = llm_config.get('provider', 'local')
    llm_model = llm_config.get('model', 'llama3.1:latest')

    # Get all general claims
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, claim_text
                FROM {schema}.ditwah_claims
                WHERE result_version_id = %s
                ORDER BY claim_order
            """, (args.version_id,))
            general_claims = cur.fetchall()

    total_stance_records = 0

    for i, claim in enumerate(general_claims):
        claim_id = claim['id']
        claim_text = claim['claim_text']

        logger.info(f"\nProcessing claim {i+1}/{len(general_claims)}: {claim_text[:60]}...")

        # Get articles for this claim
        articles = get_articles_for_general_claim(claim_id)
        logger.info(f"  Found {len(articles)} articles")

        if not articles:
            logger.warning(f"  No articles found for claim {claim_id}")
            continue

        # Analyze stance
        count = analyze_claim_stance(
            llm=llm,
            claim_id=claim_id,
            claim_text=claim_text,
            articles=articles,
            config=stance_config,
            llm_provider=llm_provider,
            llm_model=llm_model
        )

        total_stance_records += count
        logger.info(f"  ✅ Analyzed {count} articles")

    logger.info(f"\n✅ Total stance records created: {total_stance_records}")

    # Update pipeline status
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                UPDATE {schema}.result_versions
                SET pipeline_status = jsonb_set(
                    jsonb_set(
                        COALESCE(pipeline_status, '{{}}'::jsonb),
                        '{{ditwah_sentiment}}',
                        'true'::jsonb
                    ),
                    '{{ditwah_stance}}',
                    'true'::jsonb
                ),
                is_complete = true
                WHERE id = %s
            """, (args.version_id,))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"General claims: {len(general_claims)}")
    logger.info(f"Sentiment records: {sentiment_count}")
    logger.info(f"Stance records: {total_stance_records}")
    logger.info("")
    logger.info("✅ Sentiment & stance analysis complete!")
    logger.info("")
    logger.info("The pipeline is now complete. View results in the dashboard:")
    logger.info("  streamlit run dashboard/app.py")
    logger.info("")
    logger.info("Navigate to the 'Ditwah Claims' tab and select this version.")


if __name__ == '__main__':
    main()
