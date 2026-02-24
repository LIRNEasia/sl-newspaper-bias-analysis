#!/usr/bin/env python3
"""
Re-cluster existing individual claims into tighter, more specific general claims.

Copies individual claims from an existing version (skipping the expensive LLM step),
re-clusters them with tighter settings (target ~25 articles per cluster, min 15),
generates new general claim text per cluster, then runs sentiment + stance analysis.

Usage:
    # Step 1: create a new version in the dashboard (Ditwah Claims page → Create Version)
    # Step 2: run this script:

    python3 scripts/ditwah_claims/05_recluster_and_restance.py \\
        --source-version-id 518c2a5a-9340-4d73-805e-c55facabd642 \\
        --target-version-id <new-version-uuid> \\
        --target-cluster-size 25 \\
        --min-articles 15

Prerequisites:
    - Source version must have individual claims in ditwah_article_claims
    - Target version must exist (create via dashboard or versions CLI)
    - Ollama / local LLM must be running (for general claim text + stance analysis)
    - Sentiment analysis must have been run on Ditwah articles (for claim_sentiment)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_db
from src.llm import get_llm
from src.ditwah_claims import (
    cluster_individual_claims,
    generate_general_claim_from_cluster,
    store_general_claims_and_link,
    update_claim_article_counts,
    get_articles_for_general_claim,
    link_sentiment_to_general_claims,
    analyze_claim_stance,
    analyze_claim_stance_nli,
)
from src.versions import get_version

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

SEP = "=" * 80


def copy_individual_claims(source_version_id: str, target_version_id: str) -> int:
    """Copy individual claims from source version into target version."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Check source has claims
            cur.execute(f"""
                SELECT COUNT(*) AS c
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
            """, (source_version_id,))
            count = cur.fetchone()["c"]
            if count == 0:
                return 0

            # Check target doesn't already have claims
            cur.execute(f"""
                SELECT COUNT(*) AS c
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
            """, (target_version_id,))
            existing = cur.fetchone()["c"]
            if existing > 0:
                logger.info(f"Target already has {existing} individual claims — skipping copy")
                return existing

            # Copy
            cur.execute(f"""
                INSERT INTO {schema}.ditwah_article_claims
                    (article_id, result_version_id, claim_text, llm_provider, llm_model)
                SELECT article_id, %s, claim_text, llm_provider, llm_model
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
            """, (target_version_id, source_version_id))

            cur.execute(f"""
                SELECT COUNT(*) AS c
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
            """, (target_version_id,))
            return cur.fetchone()["c"]


def main():
    parser = argparse.ArgumentParser(
        description="Re-cluster existing claims into tighter general claims + run stance"
    )
    parser.add_argument(
        "--source-version-id", required=True,
        help="Version ID that already has individual claims (e.g. ditwah-v1)"
    )
    parser.add_argument(
        "--target-version-id", required=True,
        help="New version ID to store the re-clustered general claims into"
    )
    parser.add_argument(
        "--target-cluster-size", type=int, default=35,
        help="Target average articles per cluster (default: 35)"
    )
    parser.add_argument(
        "--min-articles", type=int, default=20,
        help="Minimum articles per cluster; smaller clusters merged into nearest (default: 20)"
    )
    parser.add_argument(
        "--max-clusters", type=int, default=60,
        help="Hard upper bound on number of clusters (default: 60)"
    )
    parser.add_argument(
        "--sentiment-model", type=str, default="roberta",
        help="Sentiment model to link (default: roberta)"
    )
    parser.add_argument(
        "--skip-stance", action="store_true",
        help="Skip stance analysis (useful for a quick test of clustering only)"
    )
    parser.add_argument(
        "--stance-method", type=str, default="nli",
        choices=["nli", "llm"],
        help="Stance detection method: nli (default, uses roberta-large-mnli) "
             "or llm (legacy LLM-based)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate versions
    # -----------------------------------------------------------------------
    logger.info(SEP)
    logger.info("Re-cluster & Re-stance Pipeline")
    logger.info(SEP)

    source_version = get_version(args.source_version_id)
    if not source_version:
        logger.error(f"Source version not found: {args.source_version_id}")
        sys.exit(1)

    target_version = get_version(args.target_version_id)
    if not target_version:
        logger.error(f"Target version not found: {args.target_version_id}")
        sys.exit(1)

    logger.info(f"Source version : {source_version['name']} ({args.source_version_id})")
    logger.info(f"Target version : {target_version['name']} ({args.target_version_id})")
    logger.info(f"Target cluster size : ~{args.target_cluster_size} articles each")
    logger.info(f"Minimum per cluster : {args.min_articles} articles")

    target_config  = target_version["configuration"]
    llm_config     = target_config.get("llm", source_version["configuration"].get("llm", {}))
    clustering_cfg = target_config.get("clustering", {})
    generation_cfg = target_config.get("generation", {})
    stance_cfg     = target_config.get("stance", {})

    # -----------------------------------------------------------------------
    # Step 1: Copy individual claims from source → target
    # -----------------------------------------------------------------------
    logger.info(f"\n{SEP}")
    logger.info("Step 1: Copy individual claims")
    logger.info(SEP)

    copied = copy_individual_claims(args.source_version_id, args.target_version_id)
    if copied == 0:
        logger.error("Source version has no individual claims. Run 02_generate_individual_claims.py first.")
        sys.exit(1)

    logger.info(f"✅ {copied} individual claims in target version")

    # -----------------------------------------------------------------------
    # Step 2: Cluster with new settings
    # -----------------------------------------------------------------------
    logger.info(f"\n{SEP}")
    logger.info("Step 2: Cluster individual claims")
    logger.info(SEP)

    clusters = cluster_individual_claims(
        version_id=args.target_version_id,
        config=clustering_cfg,
        max_clusters=args.max_clusters,
        target_cluster_size=args.target_cluster_size,
        min_articles=args.min_articles,
    )

    if not clusters:
        logger.error("No clusters produced.")
        sys.exit(1)

    logger.info(f"✅ {len(clusters)} clusters produced")

    # -----------------------------------------------------------------------
    # Step 3: Generate general claim text for each cluster
    # -----------------------------------------------------------------------
    logger.info(f"\n{SEP}")
    logger.info("Step 3: Generate general claim text per cluster (LLM)")
    logger.info(SEP)
    logger.info(f"Provider: {llm_config.get('provider', 'local')}  "
                f"Model: {llm_config.get('model', 'llama3.1:latest')}")
    logger.info(f"Processing {len(clusters)} clusters…")

    llm          = get_llm(llm_config)
    llm_provider = llm_config.get("provider", "local")
    llm_model    = llm_config.get("model", "llama3.1:latest")

    general_claims_data = []
    for i, cluster in enumerate(clusters):
        logger.info(f"  Cluster {i+1}/{len(clusters)}  ({len(cluster)} articles)…")
        gc = generate_general_claim_from_cluster(
            llm=llm,
            individual_claim_ids=cluster,
            version_id=args.target_version_id,
            config=generation_cfg,
        )
        general_claims_data.append(gc)
        if gc:
            logger.info(f"    → {gc['claim_text'][:90]}")
        else:
            logger.warning(f"    ⚠️  Failed for cluster {i+1}")

    successful = [g for g in general_claims_data if g]
    logger.info(f"✅ {len(successful)}/{len(clusters)} general claims generated")

    # -----------------------------------------------------------------------
    # Step 4: Store general claims + link individual claims
    # -----------------------------------------------------------------------
    logger.info(f"\n{SEP}")
    logger.info("Step 4: Store general claims and link articles")
    logger.info(SEP)

    general_claim_ids = store_general_claims_and_link(
        version_id=args.target_version_id,
        clusters=clusters,
        general_claims_data=general_claims_data,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    update_claim_article_counts(args.target_version_id)
    logger.info(f"✅ Stored {len(general_claim_ids)} general claims")

    # Verify article counts
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    MIN(article_count) AS min_arts,
                    MAX(article_count) AS max_arts,
                    AVG(article_count) AS avg_arts,
                    COUNT(*) AS total_claims
                FROM {schema}.ditwah_claims
                WHERE result_version_id = %s
            """, (args.target_version_id,))
            stats = cur.fetchone()
            logger.info(
                f"Article counts per claim — "
                f"min:{stats['min_arts']}  max:{stats['max_arts']}  "
                f"avg:{float(stats['avg_arts']):.1f}  total claims:{stats['total_claims']}"
            )

    if args.skip_stance:
        logger.info("\n⏭️  Skipping stance analysis (--skip-stance)")
        logger.info("Run 04_analyze_sentiment_stance.py when ready.")
        _finish(args.target_version_id, stance=False)
        return

    # -----------------------------------------------------------------------
    # Step 5: Link existing sentiment data
    # -----------------------------------------------------------------------
    logger.info(f"\n{SEP}")
    logger.info("Step 5: Link sentiment data to claims")
    logger.info(SEP)

    sentiment_count = link_sentiment_to_general_claims(
        version_id=args.target_version_id,
        sentiment_model=args.sentiment_model,
    )
    logger.info(f"✅ Linked {sentiment_count} sentiment records")

    # -----------------------------------------------------------------------
    # Step 6: Stance analysis
    # -----------------------------------------------------------------------
    logger.info(f"\n{SEP}")
    logger.info(f"Step 6: Stance analysis (method: {args.stance_method})")
    logger.info(SEP)

    if args.stance_method == "nli":
        from src.nli_stance import NLIStanceAnalyzer
        nli_analyzer = NLIStanceAnalyzer()
        logger.info(f"NLI model: {nli_analyzer.MODEL_NAME}")
    else:
        nli_analyzer = None
        logger.info("This is the long step — estimated time depends on cluster size × number of claims.")
        logger.info(f"Approx LLM calls: {len(general_claim_ids)} claims × avg_articles / 5 per batch")

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, claim_text
                FROM {schema}.ditwah_claims
                WHERE result_version_id = %s
                ORDER BY claim_order
            """, (args.target_version_id,))
            all_claims = cur.fetchall()

    total_stance = 0
    t0 = time.time()

    for i, claim in enumerate(all_claims):
        claim_id   = claim["id"]
        claim_text = claim["claim_text"]

        logger.info(f"\n  Claim {i+1}/{len(all_claims)}: {claim_text[:70]}…")

        articles = get_articles_for_general_claim(claim_id)
        logger.info(f"    Articles: {len(articles)}")

        if not articles:
            logger.warning("    No articles — skipping")
            continue

        if args.stance_method == "nli":
            count = analyze_claim_stance_nli(
                analyzer=nli_analyzer,
                claim_id=claim_id,
                claim_text=claim_text,
                articles=articles,
            )
        else:
            count = analyze_claim_stance(
                llm=llm,
                claim_id=claim_id,
                claim_text=claim_text,
                articles=articles,
                config=stance_cfg,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
        total_stance += count

        elapsed = time.time() - t0
        remaining_claims = len(all_claims) - (i + 1)
        avg_per_claim = elapsed / (i + 1)
        eta_min = remaining_claims * avg_per_claim / 60
        logger.info(f"    ✅ {count} stance records | ETA: {eta_min:.0f} min remaining")

    logger.info(f"\n✅ Total stance records: {total_stance}")

    _finish(args.target_version_id, stance=True)


def _finish(version_id: str, stance: bool):
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            status_update = """
                jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            COALESCE(pipeline_status, '{}'::jsonb),
                            '{ditwah_individual_claims}', 'true'::jsonb
                        ),
                        '{ditwah_general_claims}', 'true'::jsonb
                    ),
                    '{ditwah_sentiment}', 'true'::jsonb
                )
            """
            if stance:
                status_update = f"""
                    jsonb_set(
                        {status_update},
                        '{{ditwah_stance}}', 'true'::jsonb
                    )
                """
            cur.execute(f"""
                UPDATE {schema}.result_versions
                SET pipeline_status = {status_update},
                    is_complete = {'true' if stance else 'false'}
                WHERE id = %s
            """, (version_id,))

    logger.info(f"\n{SEP}")
    logger.info("DONE")
    logger.info(SEP)
    logger.info("View results in the dashboard → Ditwah Claims page → select the new version.")


if __name__ == "__main__":
    main()
