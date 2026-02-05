#!/usr/bin/env python3
"""Quick test of claims pipeline with just 10 articles."""

import sys
import os
sys.path.insert(0, '/home/ranushi/Taf_claude/sl-newspaper-bias-analysis')

from src.ditwah_claims import filter_ditwah_articles, generate_claims_from_articles
from src.llm import get_llm
from src.versions import get_version
import pandas as pd
from uuid import UUID

version_id = '7d2eb7bd-c0a2-436b-bce5-d804939f9293'
version = get_version(version_id)
config = version['configuration'].get('ditwah_claims', {})

print("=" * 60)
print("QUICK TEST: Claims Pipeline")
print("=" * 60)

# 1. Get articles (limit to 10 for testing)
print("\n1. Filtering Ditwah articles...")
articles = filter_ditwah_articles()
print(f"   Total articles: {len(articles)}")

# Limit to 10 for quick test
articles = articles[:10]
print(f"   Testing with: {len(articles)} articles\n")

# 2. Generate claims (ask for just 3 claims)
print("2. Generating claims from articles...")
llm_config = config.get('llm', {})
llm = get_llm(llm_config)

generation_config = config.get('generation', {})
generation_config['num_claims'] = 3  # Override to just 3 claims for testing

claims = generate_claims_from_articles(llm, articles, generation_config)
print(f"   Generated {len(claims)} claims\n")

# 3. Display claims
print("3. Claims generated:")
for i, claim in enumerate(claims, 1):
    print(f"   {i}. [{claim['claim_category']}] {claim['claim_text']}")
print()

# 4. Create dataframe
print("4. Creating claims dataframe...")
claims_df = pd.DataFrame([
    {
        'claim_order': i + 1,
        'claim_text': claim['claim_text'],
        'claim_category': claim['claim_category'],
        'confidence': claim.get('confidence', 0.0),
        'result_version_id': str(version_id),
        'llm_provider': llm_config.get('provider', 'local'),
        'llm_model': llm_config.get('model', 'llama3.1:latest')
    }
    for i, claim in enumerate(claims)
])

print(f"   Dataframe shape: {claims_df.shape}")
print(f"\n   Preview:")
print(claims_df[['claim_order', 'claim_category', 'claim_text']].to_string())

# 5. Save to CSV
output_dir = "/tmp/claude-1014/-home-ranushi-Taf-claude-sl-newspaper-bias-analysis/6f0836e0-d835-4dfc-9e93-d2fc8002ee15/scratchpad/ditwah_claims_output"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"test_claims_{version_id}.csv")
claims_df.to_csv(output_file, index=False)

print(f"\n5. ✅ Saved to: {output_file}")
print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
