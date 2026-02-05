#!/usr/bin/env python3
"""
Extract Claims from Ditwah Cyclone Articles using Mistral AI

This script demonstrates how to:
1. Filter articles about Cyclone Ditwah (is_ditwah_cyclone = 1)
2. Use Mistral AI to extract meaningful claims
3. Generate claims like "how government took actions on the disaster"

Usage:
    export MISTRAL_API_KEY=your_key_here
    python3 extract_ditwah_claims_mistral.py
"""

import json
import logging
from dotenv import load_dotenv

from src.db import get_db
from src.llm import MistralLLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_ditwah_articles():
    """
    Get all articles about Cyclone Ditwah from the database.

    Returns:
        List of article dictionaries with keys: id, title, content, source_id, date_posted, url
    """
    logger.info("Filtering Ditwah articles from database...")

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    id,
                    title,
                    content,
                    source_id,
                    date_posted,
                    url
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
                ORDER BY date_posted ASC
            """)
            articles = cur.fetchall()

    logger.info(f"Found {len(articles)} Ditwah articles")
    return articles


def extract_claims_with_mistral(articles, num_claims=15):
    """
    Use Mistral AI to extract claims from Ditwah articles.

    Claims include things like:
    - How government took actions on the disaster
    - Humanitarian aid provided
    - Infrastructure damage reports
    - Economic impact assessments

    Args:
        articles: List of article dictionaries
        num_claims: Number of claims to extract (default: 15)

    Returns:
        List of claim dictionaries with keys: claim_text, claim_category, confidence
    """
    logger.info(f"Using Mistral AI to extract {num_claims} claims from {len(articles)} articles...")

    # Initialize Mistral LLM
    llm = MistralLLM(
        model="mistral-large-latest",
        temperature=0.2,  # Slightly creative but mostly factual
        max_tokens=4096
    )

    # Prepare article summaries for the LLM
    # We'll send title + first 500 characters of each article
    article_summaries = []
    for article in articles:
        summary = {
            'title': article['title'],
            'excerpt': article['content'][:500] if article['content'] else '',
            'date': str(article['date_posted']),
            'source': article['source_id']
        }
        article_summaries.append(summary)

    # Define claim categories
    categories = [
        "government_response",           # How government took action
        "humanitarian_aid",              # Relief efforts, donations, support
        "infrastructure_damage",         # Buildings, roads, power, water damage
        "economic_impact",               # Financial losses, business impact
        "international_response",        # Foreign aid, international support
        "casualties_and_displacement",   # Deaths, injuries, evacuations
        "environmental_impact",          # Environmental damage, flooding
        "recovery_efforts"               # Cleanup, rebuilding, recovery
    ]

    # Create the LLM prompt
    prompt = f"""You are analyzing {len(articles)} news articles about Cyclone Ditwah that hit Sri Lanka.

Your task is to identify {num_claims} specific, verifiable claims that appear across these articles.

Articles:
{json.dumps(article_summaries, indent=2)}

Instructions:
1. Extract {num_claims} key claims or factual statements from these articles
2. Focus on claims that are:
   - Specific and verifiable (not vague opinions)
   - Mentioned across multiple articles (at least 3-4 articles)
   - Important for understanding the cyclone's impact or the response to it
3. Include claims about:
   - How the government took actions on the disaster
   - What humanitarian aid was provided
   - Infrastructure damage and casualties
   - Economic impact and recovery efforts
   - International response and support
4. Categorize each claim using these categories: {', '.join(categories)}
5. Assign a confidence score (0.0-1.0) based on how widely the claim is mentioned

Return a JSON array with this exact structure:
[
  {{
    "claim_text": "The exact claim or statement (e.g., 'The government deployed 500 military personnel for rescue operations')",
    "claim_category": "one of the categories above",
    "confidence": 0.85
  }}
]

Examples of good claims:
- "The government established emergency relief centers in affected districts"
- "UN allocated $4.5 million for disaster relief"
- "Over 10,000 families were displaced by flooding"
- "Major roads and bridges were damaged in the southern provinces"

Return ONLY the JSON array, no other text."""

    # Call Mistral AI
    logger.info("Calling Mistral AI API...")
    try:
        response = llm.generate(
            prompt=prompt,
            json_mode=True  # Ensures structured JSON output
        )

        logger.info(f"API call successful")
        logger.info(f"Tokens used: {response.usage['input_tokens']} input, {response.usage['output_tokens']} output")

        # Parse the JSON response
        claims = json.loads(response.content)

        if not isinstance(claims, list):
            logger.error("Response is not a list")
            return []

        logger.info(f"Successfully extracted {len(claims)} claims")
        return claims

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response content: {response.content[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Error calling Mistral API: {e}")
        return []


def print_claims(claims):
    """Pretty print the extracted claims."""
    print("\n" + "="*80)
    print("EXTRACTED CLAIMS FROM DITWAH CYCLONE ARTICLES")
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
        print(f"\n📌 {category.upper().replace('_', ' ')}")
        print("-" * 80)
        for i, claim in enumerate(category_claims, 1):
            print(f"{i}. {claim['claim_text']}")
            print(f"   Confidence: {claim['confidence']:.2f}")
            print()


def save_claims_to_json(claims, filename='ditwah_claims_output.json'):
    """Save claims to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(claims, f, indent=2, ensure_ascii=False)
    logger.info(f"Claims saved to {filename}")


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("DITWAH CLAIMS EXTRACTION WITH MISTRAL AI")
    logger.info("="*80)

    # Step 1: Filter Ditwah articles from database
    articles = filter_ditwah_articles()

    if not articles:
        logger.error("No Ditwah articles found in database")
        logger.error("Make sure articles are marked with is_ditwah_cyclone = 1")
        return

    # Step 2: Extract claims using Mistral AI
    claims = extract_claims_with_mistral(articles, num_claims=15)

    if not claims:
        logger.error("No claims extracted")
        return

    # Step 3: Display results
    print_claims(claims)

    # Step 4: Save to JSON file (optional)
    save_claims_to_json(claims)

    logger.info("="*80)
    logger.info("✅ COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
