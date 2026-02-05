#!/usr/bin/env python3
"""
Extract Claims from ALL Ditwah Articles Using Batching

This version splits the 1,657 articles into batches to avoid rate limits.
"""

import json
import time
from dotenv import load_dotenv
from src.db import get_db
from src.llm import MistralLLM

load_dotenv()


def get_ditwah_articles():
    """Get all Ditwah articles"""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
                ORDER BY date_posted ASC
            """)
            return cur.fetchall()


def extract_claims_batched(articles, batch_size=150, claims_per_batch=5):
    """
    Extract claims by processing articles in batches.

    Args:
        articles: All articles
        batch_size: Number of articles per batch (default: 150)
        claims_per_batch: Number of claims to extract per batch (default: 5)

    Returns:
        Combined list of all claims
    """
    llm = MistralLLM(model="mistral-large-latest", temperature=0.2)

    all_claims = []
    num_batches = (len(articles) + batch_size - 1) // batch_size

    print(f"Processing {len(articles)} articles in {num_batches} batches of {batch_size}")
    print(f"Extracting {claims_per_batch} claims per batch = ~{claims_per_batch * num_batches} total claims")
    print()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(articles))
        batch = articles[start_idx:end_idx]

        print(f"Batch {batch_idx + 1}/{num_batches}: Processing articles {start_idx + 1}-{end_idx}")

        # Prepare article data (very short excerpts)
        article_data = [
            {
                'title': a['title'],
                'excerpt': a['content'][:100] if a['content'] else '',
                'source': a['source_id']
            }
            for a in batch
        ]

        # Create prompt
        prompt = f"""Analyze these {len(batch)} articles about Cyclone Ditwah and extract {claims_per_batch} key claims.

Articles:
{json.dumps(article_data, indent=2)}

Focus on claims like:
- How the government took actions on the disaster
- What humanitarian aid was provided
- Infrastructure damage and casualties
- Economic impact and recovery efforts

For each claim, provide:
1. claim_text: The specific, verifiable claim
2. claim_category: One of: government_response, humanitarian_aid, infrastructure_damage, economic_impact, international_response, casualties_and_displacement
3. confidence: How confident you are (0.0 to 1.0)

Return as JSON array:
[
  {{
    "claim_text": "...",
    "claim_category": "...",
    "confidence": 0.9
  }}
]

Return ONLY the JSON array."""

        # Add delay between batches to avoid rate limits
        if batch_idx > 0:
            wait_time = 30  # 30 seconds between batches
            print(f"  Waiting {wait_time}s before API call...")
            time.sleep(wait_time)

        # Call API with retry
        try:
            print(f"  Calling Mistral API...")
            response = llm.generate(prompt, json_mode=True)
            batch_claims = json.loads(response.content)

            print(f"  ✅ Extracted {len(batch_claims)} claims")
            print(f"  Tokens: {response.usage['input_tokens']} in, {response.usage['output_tokens']} out")

            all_claims.extend(batch_claims)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            if "rate" in str(e).lower():
                print(f"  Rate limit hit. Waiting 120s...")
                time.sleep(120)
                # Retry once
                try:
                    response = llm.generate(prompt, json_mode=True)
                    batch_claims = json.loads(response.content)
                    print(f"  ✅ Retry successful: {len(batch_claims)} claims")
                    all_claims.extend(batch_claims)
                except Exception as e2:
                    print(f"  ❌ Retry failed: {e2}")
                    continue

        print()

    return all_claims


def deduplicate_claims(claims):
    """Remove duplicate or very similar claims."""
    unique_claims = []
    seen_texts = set()

    for claim in claims:
        # Simple deduplication by exact text match
        text_lower = claim['claim_text'].lower()
        if text_lower not in seen_texts:
            seen_texts.add(text_lower)
            unique_claims.append(claim)

    print(f"Deduplicated: {len(claims)} → {len(unique_claims)} unique claims")
    return unique_claims


def main():
    print("="*80)
    print("EXTRACTING CLAIMS FROM ALL DITWAH ARTICLES (BATCHED)")
    print("="*80)
    print()

    # Get all articles
    print("Fetching Ditwah articles from database...")
    articles = get_ditwah_articles()
    print(f"Found {len(articles)} articles")
    print()

    # Extract claims in batches
    claims = extract_claims_batched(
        articles,
        batch_size=150,        # Articles per batch
        claims_per_batch=5     # Claims per batch
    )

    if not claims:
        print("No claims extracted")
        return

    # Deduplicate
    claims = deduplicate_claims(claims)

    # Sort by confidence
    claims.sort(key=lambda x: x['confidence'], reverse=True)

    # Display results
    print("\n" + "="*80)
    print(f"EXTRACTED {len(claims)} CLAIMS")
    print("="*80)

    # Group by category
    by_category = {}
    for claim in claims:
        category = claim['claim_category']
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(claim)

    # Print by category
    for category, category_claims in by_category.items():
        print(f"\n📌 {category.upper().replace('_', ' ')} ({len(category_claims)} claims)")
        print("-" * 80)
        for i, claim in enumerate(category_claims[:10], 1):  # Show top 10 per category
            print(f"{i}. {claim['claim_text']}")
            print(f"   Confidence: {claim['confidence']:.2f}")
            print()

    # Save to file
    filename = 'ditwah_claims_all_articles.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved all {len(claims)} claims to {filename}")


if __name__ == "__main__":
    main()
