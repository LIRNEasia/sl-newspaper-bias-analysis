#!/usr/bin/env python3
"""
Cluster Individual Claims into General Claims

This script clusters similar individual claims into general claims using embeddings.
This is step 2 of the two-step claims generation process.

Usage:
    python3 scripts/ditwah_claims/03_cluster_claims.py --version-id <uuid>

Prerequisites:
    1. Run 02_generate_individual_claims.py first

Output:
    - General claims stored in ditwah_claims table (max ~40)
    - Individual claims linked to general claims
    - article_count and individual_claims_count updated
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
    cluster_individual_claims,
    generate_general_claim_from_cluster,
    store_general_claims_and_link,
    update_claim_article_counts
)
from src.versions import get_version

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Cluster individual claims into general claims')
    parser.add_argument('--version-id', type=str, required=True,
                       help='Result version ID (UUID)')
    parser.add_argument('--max-clusters', type=int, default=40,
                       help='Maximum number of general claims to create (default: 40)')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("DITWAH Claims Clustering Pipeline")
    logger.info("=" * 80)

    # Get version and config
    version = get_version(args.version_id)
    if not version:
        logger.error(f"Version not found: {args.version_id}")
        sys.exit(1)

    logger.info(f"Version: {version['name']}")
    logger.info(f"Description: {version['description']}")
    logger.info(f"Target: Max {args.max_clusters} general claims")

    config = version['configuration']
    llm_config = config.get('llm', {})
    clustering_config = config.get('clustering', {})
    generation_config = config.get('generation', {})

    # Check if individual claims exist
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
            """, (args.version_id,))
            individual_count = cur.fetchone()['count']

    if individual_count == 0:
        logger.error("❌ No individual claims found. Run 02_generate_individual_claims.py first.")
        sys.exit(1)

    logger.info(f"✅ Found {individual_count} individual claims to cluster")

    # Check if already clustered
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.ditwah_claims
                WHERE result_version_id = %s
            """, (args.version_id,))
            existing_general = cur.fetchone()['count']

    if existing_general > 0:
        logger.warning(f"⚠️  Found {existing_general} existing general claims for this version")
        response = input("Continue and regenerate? This will overwrite existing general claims. (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Aborted by user")
            sys.exit(0)

    # Cluster individual claims
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Clustering Individual Claims")
    logger.info("=" * 80)
    logger.info("Generating embeddings and clustering...")

    clusters = cluster_individual_claims(
        version_id=args.version_id,
        config=clustering_config,
        max_clusters=args.max_clusters
    )

    if not clusters:
        logger.error("❌ No clusters generated")
        sys.exit(1)

    logger.info(f"✅ Created {len(clusters)} clusters")
    logger.info(f"Cluster sizes: min={min(len(c) for c in clusters)}, "
                f"max={max(len(c) for c in clusters)}, "
                f"avg={sum(len(c) for c in clusters)/len(clusters):.1f}")

    # Initialize LLM for general claim generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Generating General Claims from Clusters")
    logger.info("=" * 80)
    logger.info(f"Provider: {llm_config.get('provider', 'local')}")
    logger.info(f"Model: {llm_config.get('model', 'llama3.1:latest')}")

    llm = get_llm(llm_config)
    logger.info("✅ LLM initialized")
    logger.info(f"\nGenerating {len(clusters)} general claims...")
    logger.info("")

    general_claims_data = []
    for i, cluster in enumerate(clusters):
        logger.info(f"  Processing cluster {i+1}/{len(clusters)} ({len(cluster)} individual claims)...")

        general_claim = generate_general_claim_from_cluster(
            llm=llm,
            individual_claim_ids=cluster,
            version_id=args.version_id,
            config=generation_config
        )

        if general_claim:
            general_claims_data.append(general_claim)
        else:
            logger.warning(f"  Failed to generate general claim for cluster {i+1}")
            general_claims_data.append(None)

    successful_claims = [c for c in general_claims_data if c is not None]
    logger.info(f"\n✅ Generated {len(successful_claims)} general claims")

    # Store general claims and link to individual claims
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Storing General Claims and Linking")
    logger.info("=" * 80)

    llm_provider = llm_config.get('provider', 'local')
    llm_model = llm_config.get('model', 'llama3.1:latest')

    general_claim_ids = store_general_claims_and_link(
        version_id=args.version_id,
        clusters=clusters,
        general_claims_data=general_claims_data,
        llm_provider=llm_provider,
        llm_model=llm_model
    )

    logger.info(f"✅ Stored {len(general_claim_ids)} general claims")

    # Update article counts
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Updating Article Counts")
    logger.info("=" * 80)

    update_claim_article_counts(args.version_id)
    logger.info("✅ Updated article counts")

    # Update pipeline status
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                UPDATE {schema}.result_versions
                SET pipeline_status = jsonb_set(
                    COALESCE(pipeline_status, '{{}}'::jsonb),
                    '{{ditwah_general_claims}}',
                    'true'::jsonb
                )
                WHERE id = %s
            """, (args.version_id,))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Individual claims: {individual_count}")
    logger.info(f"Clusters created: {len(clusters)}")
    logger.info(f"General claims: {len(successful_claims)}")
    logger.info(f"Average claims per cluster: {individual_count/len(successful_claims):.1f}")
    logger.info("")
    logger.info("✅ Claims clustering complete!")
    logger.info("")
    logger.info("Next step: Analyze sentiment and stance")
    logger.info(f"  python3 scripts/ditwah_claims/04_analyze_sentiment_stance.py --version-id {args.version_id}")


if __name__ == '__main__':
    main()
