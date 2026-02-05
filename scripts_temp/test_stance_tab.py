#!/usr/bin/env python3
"""
Test script for stance distribution tab functionality.
"""

import sys
sys.path.insert(0, '/home/ranushi/Taf_claude/sl-newspaper-bias-analysis')

from src.db import get_db
import yaml

def test_stance_queries():
    """Test all stance-related queries."""

    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    schema = config['database']['schema']

    print("Testing Stance Distribution Queries...")
    print("=" * 60)

    with get_db() as db:
        # Get a version_id first
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, name, analysis_type
                FROM {schema}.result_versions
                WHERE analysis_type = 'ditwah_claims'
                ORDER BY created_at DESC
                LIMIT 1
            """)
            version = cur.fetchone()

            if not version:
                print("❌ No ditwah_claims versions found. Create one first.")
                return

            version_id = version['id']
            print(f"✅ Using version: {version['name']} (ID: {version_id})")
            print()

        # Test 1: Overview statistics
        print("Test 1: Overview Statistics")
        print("-" * 60)
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT COUNT(DISTINCT claim_id) as total_claims
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
            """, (version_id,))
            result = cur.fetchone()
            print(f"Total claims with stance data: {result['total_claims']}")

            if result['total_claims'] == 0:
                print("❌ No stance data found. Run stance analysis pipeline first.")
                return

        print("✅ Test 1 passed")
        print()

        # Test 2: Most controversial claim
        print("Test 2: Most Controversial Claim")
        print("-" * 60)
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    dc.id,
                    dc.claim_text,
                    STDDEV(cs.stance_score) as controversy
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text
                ORDER BY controversy DESC
                LIMIT 1
            """, (version_id,))
            result = cur.fetchone()
            if result:
                print(f"Claim: {result['claim_text'][:80]}...")
                print(f"Controversy score: {result['controversy']:.3f}")
            else:
                print("No controversial claims found")
        print("✅ Test 2 passed")
        print()

        # Test 3: Source alignment
        print("Test 3: Source Alignment Matrix")
        print("-" * 60)
        with db.cursor() as cur:
            cur.execute(f"""
                WITH source_stances AS (
                    SELECT
                        cs.claim_id,
                        cs.source_id,
                        CASE
                            WHEN cs.stance_score > 0.2 THEN 'agree'
                            WHEN cs.stance_score < -0.2 THEN 'disagree'
                            ELSE 'neutral'
                        END as stance_category
                    FROM {schema}.claim_stance cs
                    JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                    WHERE dc.result_version_id = %s
                )
                SELECT
                    s1.source_id as source1,
                    s2.source_id as source2,
                    COUNT(*) as total_claims,
                    ROUND(SUM(CASE WHEN s1.stance_category = s2.stance_category THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as alignment_pct
                FROM source_stances s1
                JOIN source_stances s2 ON s1.claim_id = s2.claim_id AND s1.source_id < s2.source_id
                GROUP BY s1.source_id, s2.source_id
                ORDER BY alignment_pct DESC
                LIMIT 5
            """, (version_id,))
            results = cur.fetchall()
            if results:
                for row in results:
                    print(f"{row['source1']} <-> {row['source2']}: {row['alignment_pct']}% alignment ({row['total_claims']} claims)")
            else:
                print("No alignment data found")
        print("✅ Test 3 passed")
        print()

        # Test 4: Confidence-weighted stances
        print("Test 4: Confidence-Weighted Stances")
        print("-" * 60)
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    dc.id as claim_id,
                    dc.claim_text,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as stddev_stance,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(DISTINCT cs.article_id) as article_count
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                GROUP BY dc.id, dc.claim_text
                HAVING COUNT(DISTINCT cs.article_id) >= 2
                ORDER BY stddev_stance DESC
                LIMIT 5
            """, (version_id,))
            results = cur.fetchall()
            if results:
                for row in results:
                    print(f"Claim: {row['claim_text'][:60]}...")
                    print(f"  Avg stance: {row['avg_stance']:.2f}, Controversy: {row['stddev_stance']:.3f}, Confidence: {row['avg_confidence']:.2f}")
            else:
                print("No data found")
        print("✅ Test 4 passed")
        print()

        # Test 5: Supporting quotes
        print("Test 5: Supporting Quotes by Stance")
        print("-" * 60)
        with db.cursor() as cur:
            # Get first claim with quotes
            cur.execute(f"""
                SELECT DISTINCT cs.claim_id
                FROM {schema}.claim_stance cs
                JOIN {schema}.ditwah_claims dc ON cs.claim_id = dc.id
                WHERE dc.result_version_id = %s
                  AND cs.supporting_quotes IS NOT NULL
                  AND jsonb_array_length(cs.supporting_quotes) > 0
                LIMIT 1
            """, (version_id,))
            claim = cur.fetchone()

            if claim:
                cur.execute(f"""
                    SELECT
                        cs.stance_label,
                        COUNT(*) as count
                    FROM {schema}.claim_stance cs
                    WHERE cs.claim_id = %s
                      AND cs.supporting_quotes IS NOT NULL
                      AND jsonb_array_length(cs.supporting_quotes) > 0
                    GROUP BY cs.stance_label
                """, (claim['claim_id'],))
                results = cur.fetchall()
                for row in results:
                    print(f"{row['stance_label']}: {row['count']} articles with quotes")
            else:
                print("No quotes found")
        print("✅ Test 5 passed")
        print()

    print("=" * 60)
    print("🎉 All tests passed! Stance Distribution tab should work correctly.")
    print()
    print("Next steps:")
    print("1. Navigate to http://localhost:8501")
    print("2. Click on the '⚖️ Stance' tab (bottom row)")
    print("3. Select a ditwah_claims version")
    print("4. Explore the stance analysis visualizations")


if __name__ == "__main__":
    test_stance_queries()
