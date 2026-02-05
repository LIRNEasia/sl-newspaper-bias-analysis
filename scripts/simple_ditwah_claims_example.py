#!/usr/bin/env python3
"""
Simple Example: Extract Claims from Ditwah Articles with Mistral AI

This is a minimal example showing the core components.
"""

import json
import random
import time
from dotenv import load_dotenv
from src.db import get_db
from src.llm import MistralLLM

load_dotenv()


# 1. GET DITWAH ARTICLES FROM DATABASE
def get_ditwah_articles(sample_size=None):
    """
    Filter articles where is_ditwah_cyclone = 1

    Args:
        sample_size: Number of articles to sample (default: None = use all)
                    Set to a number to limit the sample
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
                ORDER BY date_posted ASC
            """)
            articles = cur.fetchall()

            # Sample if requested
            if sample_size and len(articles) > sample_size:
                print(f"Sampling {sample_size} articles from {len(articles)} total (to reduce API token usage)")
                # Use random seed for reproducibility
                random.seed(42)
                articles = random.sample(articles, sample_size)

            return articles


# 2. EXTRACT CLAIMS USING MISTRAL AI
def extract_claims(articles):
    """Use Mistral to extract claims like 'how government took actions on disaster'"""

    # Initialize Mistral
    llm = MistralLLM(model="mistral-large-latest", temperature=0.2)

    # Prepare article data (title + snippet)
    # Use VERY short excerpts to handle large dataset (1600+ articles)
    article_data = [
        {
            'title': a['title'],
            'excerpt': a['content'][:100] if a['content'] else '',  # Just 100 chars
            'source': a['source_id']
        }
        for a in articles
    ]

    print(f"Prepared {len(article_data)} articles for analysis...")

    # Create prompt for claim extraction
    prompt = f"""Analyze these {len(articles)} articles about Cyclone Ditwah and extract 15 key claims.

Articles:
{json.dumps(article_data, indent=2)}

Focus on claims like:
- How the government took actions on the disaster
- What humanitarian aid was provided
- Infrastructure damage and casualties
- Economic impact and recovery efforts

For each claim, provide:
1. claim_text: The specific claim (e.g., "Government deployed 500 troops for rescue")
2. claim_category: One of these categories:
   - government_response
   - humanitarian_aid
   - infrastructure_damage
   - economic_impact
   - international_response
   - casualties_and_displacement
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

    # Wait a bit to avoid rate limits
    print("Waiting 60 seconds to avoid rate limits...")
    time.sleep(60)

    # Call Mistral API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Calling Mistral API (attempt {attempt + 1}/{max_retries})...")
            response = llm.generate(prompt, json_mode=True)
            claims = json.loads(response.content)

            print(f"✅ Extracted {len(claims)} claims")
            print(f"Tokens used: {response.usage['input_tokens']} in, {response.usage['output_tokens']} out")

            return claims

        except Exception as e:
            if "rate" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 120 * (attempt + 1)  # 120s, 240s
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}")
                raise

    return []


# 3. MAIN EXECUTION
if __name__ == "__main__":
    print("Fetching Ditwah articles...")
    articles = get_ditwah_articles()
    print(f"Found {len(articles)} articles")

    print("\nExtracting claims with Mistral AI...")
    claims = extract_claims(articles)

    print("\n" + "="*80)
    print("CLAIMS EXTRACTED:")
    print("="*80)

    for i, claim in enumerate(claims, 1):
        print(f"\n{i}. [{claim['claim_category']}]")
        print(f"   {claim['claim_text']}")
        print(f"   Confidence: {claim['confidence']}")

    # Save to file
    with open('ditwah_claims.json', 'w') as f:
        json.dump(claims, f, indent=2)
    print(f"\n✅ Saved to ditwah_claims.json")
