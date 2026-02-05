#!/usr/bin/env python3
"""
Load claims from JSON file into the database.

This allows us to use the claims we already extracted without hitting rate limits again.
"""

import json
import sys
from uuid import UUID

from src.db import get_db
from src.versions import update_pipeline_status


def load_claims_from_json(version_id: UUID, json_file: str):
    """Load claims from JSON file into database."""

    # Read JSON file
    print(f"Reading claims from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        claims = json.load(f)

    print(f"Found {len(claims)} claims in JSON file")

    # Insert into database
    with get_db() as db:
        schema = db.config["schema"]
        inserted_count = 0

        with db.cursor() as cur:
            for i, claim in enumerate(claims, 1):
                cur.execute(f"""
                    INSERT INTO {schema}.ditwah_claims (
                        result_version_id,
                        claim_text,
                        claim_category,
                        claim_order,
                        llm_provider,
                        llm_model
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (result_version_id, claim_text) DO UPDATE
                    SET claim_category = EXCLUDED.claim_category,
                        claim_order = EXCLUDED.claim_order
                """, (
                    str(version_id),
                    claim['claim_text'],
                    claim['claim_category'],
                    i,  # claim_order
                    'mistral',
                    'mistral-large-latest'
                ))
                inserted_count += 1

                if inserted_count % 10 == 0:
                    print(f"  Inserted {inserted_count}/{len(claims)} claims...")

        print(f"✅ Inserted {inserted_count} claims into database")

    # Mark pipeline as complete
    update_pipeline_status(str(version_id), 'ditwah_claims', True)
    print("✅ Updated pipeline status")

    return inserted_count


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 load_claims_to_database.py <version-id> <json-file>")
        print("Example: python3 load_claims_to_database.py bdbab0dc-18c5-4504-95e0-e7372946e114 ditwah_claims_all_articles.json")
        sys.exit(1)

    version_id = UUID(sys.argv[1])
    json_file = sys.argv[2]

    print("="*80)
    print("LOADING CLAIMS INTO DATABASE")
    print("="*80)
    print(f"Version ID: {version_id}")
    print(f"JSON file: {json_file}")
    print()

    count = load_claims_from_json(version_id, json_file)

    print()
    print("="*80)
    print(f"✅ COMPLETE - {count} claims loaded")
    print("="*80)
    print()
    print("You can now view the claims in the dashboard!")
    print("Note: Sentiment and stance analysis are not included yet.")
    print("To add them, run the full pipeline later when rate limits reset.")
