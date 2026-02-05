#!/usr/bin/env python3
"""
Mark Ditwah Articles

Identifies articles related to Cyclone Ditwah and marks them with is_ditwah_cyclone = TRUE.
Uses keyword matching on title and content.

Usage:
    python3 scripts/ditwah_claims/01_mark_ditwah_articles.py
"""

import logging
from src.db import get_db

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mark_ditwah_articles():
    """
    Mark articles as Ditwah-related based on keyword matching.

    Updates news_articles table to set is_ditwah_cyclone = TRUE for articles
    that mention "ditwah" in title or content.
    """
    with get_db() as db:
        schema = db.config["schema"]

        # First, check current count
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
            """)
            current_count = cur.fetchone()['count']
            logger.info(f"Current Ditwah articles: {current_count}")

        # Mark articles with 'ditwah' in title or content
        with db.cursor() as cur:
            cur.execute(f"""
                UPDATE {schema}.news_articles
                SET is_ditwah_cyclone = 1
                WHERE (
                    LOWER(title) LIKE '%ditwah%'
                    OR LOWER(content) LIKE '%ditwah%'
                )
                AND is_ditwah_cyclone != 1
            """)
            updated_count = cur.rowcount

        db._conn.commit()


        # Get final count
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(*) as count
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
            """)
            final_count = cur.fetchone()['count']

        logger.info(f"Marked {updated_count} new articles as Ditwah-related")
        logger.info(f"Total Ditwah articles: {final_count}")

        # Show breakdown by source
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    source_id as source_name,
                    COUNT(*) as article_count
                FROM {schema}.news_articles n
                WHERE n.is_ditwah_cyclone = 1
                GROUP BY source_id
                ORDER BY article_count DESC
            """)
            breakdown = cur.fetchall()

        logger.info("Breakdown by source:")
        for row in breakdown:
            logger.info(f"  {row['source_name']}: {row['article_count']} articles")


if __name__ == "__main__":
    logger.info("Starting Ditwah article marking...")
    mark_ditwah_articles()
    logger.info("Complete!")
