#!/usr/bin/env python3
"""
Generate Individual Claims for DITWAH Articles

This script generates ONE specific claim for each DITWAH article using LLM.
This is step 1 of the two-step claims generation process.

Usage:
    python3 scripts/ditwah_claims/02_generate_individual_claims.py --version-id <uuid>

Prerequisites:
    1. Run 01_mark_ditwah_articles.py to mark DITWAH articles
    2. Create a ditwah_claims version
    3. Ensure Ollama is running (for local LLM)

Output:
    - Individual claims stored in ditwah_article_claims table
    - One claim per article
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
    filter_ditwah_articles,
    generate_individual_claims_batch,
    store_individual_claims
)
from src.versions import get_version

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate individual claims for DITWAH articles')
    parser.add_argument('--version-id', type=str, required=True,
                       help='Result version ID (UUID)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("DITWAH Individual Claims Generation Pipeline")
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
    generation_config = config.get('generation', {})

    # Check if already generated
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
            """, (args.version_id,))
            existing_count = cur.fetchone()['count']

    if existing_count > 0:
        logger.warning(f"⚠️  Found {existing_count} existing individual claims for this version")
        response = input("Continue and regenerate claims? This will overwrite existing claims. (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted by user")
            sys.exit(0)

    # Filter DITWAH articles
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Fetching DITWAH articles")
    logger.info("=" * 80)

    articles = filter_ditwah_articles()
    if not articles:
        logger.error("❌ No DITWAH articles found. Run 01_mark_ditwah_articles.py first.")
        sys.exit(1)

    logger.info(f"✅ Found {len(articles)} DITWAH articles")

    # Initialize LLM
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Initializing LLM")
    logger.info("=" * 80)
    logger.info(f"Provider: {llm_config.get('provider', 'local')}")
    logger.info(f"Model: {llm_config.get('model', 'llama3.1:latest')}")

    llm = get_llm(llm_config)
    logger.info("✅ LLM initialized")

    # Generate individual claims
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Generating Individual Claims")
    logger.info("=" * 80)
    logger.info(f"Processing {len(articles)} articles...")
    logger.info("This may take 1-2 hours with local LLM")
    logger.info("")

    llm_provider = llm_config.get('provider', 'local')
    llm_model = llm_config.get('model', 'llama3.1:latest')

    claims_data = generate_individual_claims_batch(
        llm=llm,
        articles=articles,
        config=llm_config,
        llm_provider=llm_provider,
        llm_model=llm_model
    )

    if not claims_data:
        logger.error("❌ No claims generated")
        sys.exit(1)

    logger.info(f"\n✅ Generated {len(claims_data)} individual claims")

    # Store to database
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Storing Claims to Database")
    logger.info("=" * 80)

    claim_ids = store_individual_claims(args.version_id, claims_data)
    logger.info(f"✅ Stored {len(claim_ids)} individual claims")

    # Update pipeline status
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                UPDATE {schema}.result_versions
                SET pipeline_status = jsonb_set(
                    COALESCE(pipeline_status, '{{}}'::jsonb),
                    '{{ditwah_individual_claims}}',
                    'true'::jsonb
                )
                WHERE id = %s
            """, (args.version_id,))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Articles processed: {len(articles)}")
    logger.info(f"Individual claims generated: {len(claims_data)}")
    logger.info(f"Success rate: {len(claims_data)/len(articles)*100:.1f}%")
    logger.info("")
    logger.info("✅ Individual claims generation complete!")
    logger.info("")
    logger.info("Next step: Run clustering to create general claims")
    logger.info(f"  python3 scripts/ditwah_claims/03_cluster_claims.py --version-id {args.version_id}")


if __name__ == '__main__':
    main()
