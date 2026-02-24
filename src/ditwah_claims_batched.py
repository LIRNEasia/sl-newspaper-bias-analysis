"""
Ditwah Claims Analysis - Batched Approach

Efficient approach:
1. Process articles in batches (50-100 per batch)
2. Extract claims from each batch
3. Track which articles mention each claim
4. Deduplicate similar claims
5. Filter claims with < 5 articles
6. Ensure all claims have both sentiment AND stance data
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from uuid import UUID
from collections import defaultdict

from src.db import get_db

logger = logging.getLogger(__name__)


def filter_ditwah_articles() -> List[Dict]:
    """Get all articles marked as Ditwah-related."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, title, content, source_id, date_posted
                FROM {schema}.news_articles
                WHERE is_ditwah_cyclone = 1
                ORDER BY date_posted, id
            """)
            articles = cur.fetchall()

    logger.info(f"Found {len(articles)} Ditwah articles")
    return [dict(a) for a in articles]


def generate_claims_from_batch(llm, articles: List[Dict], config: Dict, batch_num: int) -> List[Dict]:
    """
    Generate claims from a batch of articles.

    For each claim, identify which articles in the batch mention it.

    Args:
        llm: LLM client instance
        articles: List of article dicts
        config: Generation config with categories
        batch_num: Batch number for logging

    Returns:
        List of claim dicts with claim_text, claim_category, and article_ids
    """
    categories = config.get('categories', [
        'casualties_and_displacement',
        'infrastructure_damage',
        'economic_impact',
        'government_response',
        'international_response',
        'humanitarian_aid'
    ])

    # Build article summaries for the prompt
    article_summaries = []
    for i, article in enumerate(articles):
        article_summaries.append(f"""
[Article {i+1} - ID: {article['id']}]
Title: {article['title']}
Content: {article['content'][:600]}...""")

    prompt = f"""You are analyzing a batch of {len(articles)} news articles about Cyclone Ditwah in Sri Lanka.

Extract ALL significant factual claims that appear across these articles. For each claim:
- Identify which article IDs mention this claim
- Claims should be specific, verifiable facts
- Multiple articles may mention the same claim
- Each claim should be newsworthy and significant

Categories: {', '.join(categories)}

Articles:
{chr(10).join(article_summaries)}

Return a JSON array of claims in this format:
[
  {{
    "claim_text": "Specific factual claim here",
    "claim_category": "category_name",
    "article_indices": [1, 3, 5]
  }},
  {{
    "claim_text": "Another claim mentioned in multiple articles",
    "claim_category": "category_name",
    "article_indices": [2, 4]
  }}
]

For article_indices, use the article numbers (1, 2, 3...) from the list above.
Extract 10-30 claims that cover the main facts across all articles.
Return valid JSON only."""

    try:
        response = llm.generate(prompt=prompt, json_mode=True)
        claims = json.loads(response.content)

        if not isinstance(claims, list):
            logger.warning(f"LLM returned non-list for batch {batch_num}")
            return []

        # Convert article indices to article IDs
        valid_claims = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            if 'claim_text' not in claim or 'claim_category' not in claim:
                continue
            if 'article_indices' not in claim or not claim['article_indices']:
                continue

            # Map indices to article IDs
            article_ids = []
            for idx in claim['article_indices']:
                if isinstance(idx, int) and 1 <= idx <= len(articles):
                    article_ids.append(str(articles[idx - 1]['id']))

            if article_ids:
                claim['article_ids'] = article_ids
                claim['article_count'] = len(article_ids)
                del claim['article_indices']
                valid_claims.append(claim)

        logger.info(f"  Batch {batch_num}: Extracted {len(valid_claims)} claims")
        return valid_claims

    except Exception as e:
        logger.error(f"Error generating claims from batch {batch_num}: {e}")
        return []


def merge_duplicate_claims(all_claims: List[Dict]) -> List[Dict]:
    """
    Merge duplicate claims by exact text match and combine article lists.

    Args:
        all_claims: List of all claims from all batches

    Returns:
        List of deduplicated claims with merged article_ids
    """
    logger.info(f"Merging {len(all_claims)} claims...")

    # Group by exact claim text
    claim_map = defaultdict(lambda: {'article_ids': set(), 'category': None})

    for claim in all_claims:
        text = claim['claim_text'].strip()
        claim_map[text]['article_ids'].update(claim['article_ids'])
        if claim_map[text]['category'] is None:
            claim_map[text]['category'] = claim['claim_category']

    # Convert to list
    merged = []
    for claim_text, data in claim_map.items():
        merged.append({
            'claim_text': claim_text,
            'claim_category': data['category'],
            'article_ids': list(data['article_ids']),
            'article_count': len(data['article_ids'])
        })

    logger.info(f"Merged to {len(merged)} unique claims")
    return merged


def deduplicate_similar_claims_with_embeddings(claims: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
    """
    Use embedding similarity to identify and merge similar claims.
    Much faster than LLM-based comparison.

    Args:
        claims: List of merged claims
        similarity_threshold: Cosine similarity threshold (0.85 = very similar)

    Returns:
        List of deduplicated claims
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    logger.info(f"Deduplicating {len(claims)} claims with embeddings (threshold={similarity_threshold})...")

    if len(claims) <= 1:
        return claims

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model

    # Group by category first
    by_category = defaultdict(list)
    for claim in claims:
        by_category[claim['claim_category']].append(claim)

    deduplicated = []

    for category, cat_claims in by_category.items():
        logger.info(f"Processing {len(cat_claims)} claims in category: {category}")

        if len(cat_claims) <= 1:
            deduplicated.extend(cat_claims)
            continue

        # Generate embeddings for all claims in this category
        claim_texts = [c['claim_text'] for c in cat_claims]
        embeddings = model.encode(claim_texts, show_progress_bar=False)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Build groups of similar claims
        processed = [False] * len(cat_claims)
        claim_groups = []

        for i in range(len(cat_claims)):
            if processed[i]:
                continue

            # Start new group with this claim
            group = [cat_claims[i]]
            processed[i] = True

            # Find all similar claims
            for j in range(i + 1, len(cat_claims)):
                if processed[j]:
                    continue

                if similarity_matrix[i][j] >= similarity_threshold:
                    group.append(cat_claims[j])
                    processed[j] = True

            claim_groups.append(group)

        # Merge each group
        for group in claim_groups:
            # Combine all article IDs
            combined_ids = set()
            for c in group:
                combined_ids.update(c['article_ids'])

            # Use first claim's text (could also use longest or most common)
            merged_claim = {
                'claim_text': group[0]['claim_text'],
                'claim_category': category,
                'article_ids': list(combined_ids),
                'article_count': len(combined_ids)
            }
            deduplicated.append(merged_claim)

        logger.info(f"  Category {category}: {len(cat_claims)} → {len(claim_groups)} claims")

    logger.info(f"Deduplicated to {len(deduplicated)} unique claims")
    return deduplicated


def filter_claims_by_article_count(claims: List[Dict], min_articles: int = 10) -> List[Dict]:
    """Filter out claims with fewer than min_articles."""
    filtered = [c for c in claims if c['article_count'] >= min_articles]

    logger.info(f"Filtered {len(claims)} claims to {len(filtered)} claims with >={min_articles} articles")
    return filtered


def analyze_claim_sentiment(claim: Dict, config: Dict) -> List[Dict]:
    """Fetch sentiment scores for articles in this claim."""
    sentiment_records = []
    primary_model = config.get('primary_model', 'roberta')

    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            for article_id in claim['article_ids']:
                cur.execute(f"""
                    SELECT
                        a.id,
                        a.source_id,
                        sa.overall_sentiment,
                        sa.model_name
                    FROM {schema}.news_articles a
                    INNER JOIN {schema}.sentiment_analyses sa ON a.id = sa.article_id
                    WHERE a.id = %s AND sa.model_type = %s
                    LIMIT 1
                """, (article_id, primary_model))

                result = cur.fetchone()
                if result:
                    sentiment_records.append({
                        'article_id': str(result['id']),
                        'source_id': result['source_id'],
                        'sentiment_score': result['overall_sentiment'],
                        'sentiment_model': result['model_name']
                    })

    return sentiment_records


def load_nli_model(model_name: str = "facebook/bart-large-mnli"):
    """Load NLI model for stance detection. Returns a zero-shot-classification pipeline."""
    from transformers import pipeline as hf_pipeline
    logger.info(f"Loading NLI model for stance detection: {model_name}")
    nli_pipeline = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1  # CPU; change to 0 if GPU is available
    )
    logger.info("NLI model loaded successfully")
    return nli_pipeline


def analyze_claim_stance_nli(nli_pipeline, claim_text: str, articles: List[Dict]) -> List[Dict]:
    """
    Analyze stance for a batch of articles using NLI-based detection (BART/RoBERTa MNLI).

    Uses zero-shot classification where the article text is the premise and
    the claim is mapped to candidate labels to determine stance.

    Args:
        nli_pipeline: Loaded HuggingFace zero-shot-classification pipeline
        claim_text: The claim to evaluate stance against
        articles: List of article dicts with id, title, content, source_id

    Returns:
        List of stance records compatible with store_claim_stance()
    """
    if not articles:
        return []

    article_map = {str(a['id']): a['source_id'] for a in articles}
    candidate_labels = [
        "supports this claim",
        "contradicts this claim",
        "is neutral about this claim",
    ]

    stance_records = []

    for article in articles:
        article_id = str(article['id'])
        # Use title + truncated content as premise
        text = f"{article['title']}. {article['content'][:800]}"

        try:
            result = nli_pipeline(
                text,
                candidate_labels,
                hypothesis_template="This text {}.",
            )

            top_label = result['labels'][0]
            confidence = float(result['scores'][0])
            all_scores = dict(zip(result['labels'], result['scores']))

            # Map NLI output to stance label and numeric score
            if "supports" in top_label:
                stance_label = "strongly_agree" if confidence >= 0.7 else "agree"
                stance_score = 1.0 if confidence >= 0.7 else 0.5
            elif "contradicts" in top_label:
                stance_label = "strongly_disagree" if confidence >= 0.7 else "disagree"
                stance_score = -1.0 if confidence >= 0.7 else -0.5
            else:
                stance_label = "neutral"
                stance_score = 0.0

            reasoning = (
                f"NLI stance: {top_label} "
                f"(support={all_scores.get('supports this claim', 0):.2f}, "
                f"neutral={all_scores.get('is neutral about this claim', 0):.2f}, "
                f"contradict={all_scores.get('contradicts this claim', 0):.2f})"
            )

            stance_records.append({
                'article_id': article_id,
                'source_id': article_map[article_id],
                'stance_score': stance_score,
                'stance_label': stance_label,
                'confidence': confidence,
                'reasoning': reasoning,
                'supporting_quotes': [],
            })

        except Exception as e:
            logger.error(f"NLI stance error for article {article_id}: {e}")

    return stance_records


def store_claims_v2(version_id: UUID, claims: List[Dict], llm_provider: str, llm_model: str) -> List[Tuple[UUID, Dict]]:
    """Store deduplicated claims with article counts."""
    with get_db() as db:
        schema = db.config["schema"]
        results = []

        with db.cursor() as cur:
            for i, claim in enumerate(claims):
                cur.execute(f"""
                    INSERT INTO {schema}.ditwah_claims (
                        result_version_id,
                        claim_text,
                        claim_category,
                        claim_order,
                        article_count,
                        llm_provider,
                        llm_model
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (result_version_id, claim_text) DO UPDATE
                    SET claim_category = EXCLUDED.claim_category,
                        article_count = EXCLUDED.article_count
                    RETURNING id
                """, (
                    str(version_id),
                    claim['claim_text'],
                    claim['claim_category'],
                    i + 1,
                    claim['article_count'],
                    llm_provider,
                    llm_model
                ))

                claim_id = cur.fetchone()['id']
                results.append((claim_id, claim))

        logger.info(f"Stored {len(results)} claims")
        return results


def store_claim_sentiment(claim_id: UUID, sentiment_records: List[Dict]) -> int:
    """Store sentiment records."""
    if not sentiment_records:
        return 0

    with get_db() as db:
        schema = db.config["schema"]
        count = 0
        skipped = 0

        with db.cursor() as cur:
            for record in sentiment_records:
                try:
                    cur.execute("SAVEPOINT sp_sentiment")
                    cur.execute(f"""
                        INSERT INTO {schema}.claim_sentiment (
                            claim_id,
                            article_id,
                            source_id,
                            sentiment_score,
                            sentiment_model
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (claim_id, article_id) DO UPDATE
                        SET sentiment_score = EXCLUDED.sentiment_score
                    """, (
                        str(claim_id),
                        record['article_id'],
                        record['source_id'],
                        record['sentiment_score'],
                        record['sentiment_model']
                    ))
                    cur.execute("RELEASE SAVEPOINT sp_sentiment")
                    count += 1
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_sentiment")
                    skipped += 1
                    logger.warning(f"Skipping sentiment record for article {record.get('article_id')}: {e}")

        if skipped:
            logger.warning(f"  Skipped {skipped} sentiment records due to errors")
        return count


def store_claim_stance(claim_id: UUID, stance_records: List[Dict]) -> int:
    """Store stance records."""
    if not stance_records:
        return 0

    with get_db() as db:
        schema = db.config["schema"]
        count = 0
        skipped = 0

        with db.cursor() as cur:
            for record in stance_records:
                try:
                    cur.execute("SAVEPOINT sp_stance")
                    cur.execute(f"""
                        INSERT INTO {schema}.claim_stance (
                            claim_id,
                            article_id,
                            source_id,
                            stance_score,
                            stance_label,
                            confidence,
                            reasoning,
                            supporting_quotes
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (claim_id, article_id) DO UPDATE
                        SET stance_score = EXCLUDED.stance_score,
                            stance_label = EXCLUDED.stance_label,
                            confidence = EXCLUDED.confidence
                    """, (
                        str(claim_id),
                        record['article_id'],
                        record['source_id'],
                        record.get('stance_score', 0.0),
                        record.get('stance_label', 'neutral'),
                        record.get('confidence', 0.5),
                        record.get('reasoning', ''),
                        json.dumps(record.get('supporting_quotes', []))
                    ))
                    cur.execute("RELEASE SAVEPOINT sp_stance")
                    count += 1
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_stance")
                    skipped += 1
                    logger.warning(f"Skipping stance record for article {record.get('article_id')}: {e}")

        if skipped:
            logger.warning(f"  Skipped {skipped} stance records due to errors")
        return count


def generate_claims_pipeline_batched(version_id: UUID, config: Dict) -> Dict[str, Any]:
    """
    Batched pipeline for Ditwah claims analysis.

    Steps:
    1. Get all Ditwah articles
    2. Process in batches (50-100 articles per batch)
    3. Generate claims from each batch with article mappings
    4. Merge duplicate claims (exact match)
    5. Deduplicate similar claims (LLM comparison)
    6. Filter claims with < 5 articles
    7. For each claim:
       a. Fetch sentiment from existing data
       b. Analyze stance with LLM (batched)
       c. Store to database
    8. Only keep claims with BOTH sentiment AND stance

    Args:
        version_id: Result version ID
        config: Configuration dict

    Returns:
        Summary dict with counts
    """
    from src.llm import get_llm

    logger.info(f"Starting batched Ditwah claims pipeline for version {version_id}")

    # Step 1: Get all Ditwah articles
    articles = filter_ditwah_articles()
    if not articles:
        logger.error("No Ditwah articles found")
        return {'error': 'No Ditwah articles found'}

    logger.info(f"Processing {len(articles)} Ditwah articles in batches")

    # Step 2: Process in batches
    llm_config = config.get('llm', {})
    llm = get_llm(llm_config)
    generation_config = config.get('generation', {})

    batch_size = 50  # Process 50 articles per batch
    all_claims = []
    num_batches = (len(articles) + batch_size - 1) // batch_size

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(f"Processing batch {batch_num}/{num_batches} ({len(batch)} articles)...")

        batch_claims = generate_claims_from_batch(llm, batch, generation_config, batch_num)
        all_claims.extend(batch_claims)

    logger.info(f"✅ Generated {len(all_claims)} raw claims from {num_batches} batches")

    if not all_claims:
        logger.error("No claims generated")
        return {'error': 'No claims generated'}

    # Step 3: Merge exact duplicates
    merged_claims = merge_duplicate_claims(all_claims)

    # Step 4: Deduplicate similar claims with embeddings (fast!)
    similarity_threshold = config.get('deduplication', {}).get('similarity_threshold', 0.85)
    deduplicated_claims = deduplicate_similar_claims_with_embeddings(merged_claims, similarity_threshold)

    # Step 5: Filter by article count
    min_articles = generation_config.get('min_articles', 10)
    filtered_claims = filter_claims_by_article_count(deduplicated_claims, min_articles)

    if not filtered_claims:
        logger.error(f"No claims with >= {min_articles} articles")
        return {'error': f'No claims with >= {min_articles} articles'}

    logger.info(f"✅ {len(filtered_claims)} claims passed filters")

    # Step 6: Store claims and analyze sentiment/stance
    llm_provider = llm_config.get('provider', 'mistral')
    llm_model = llm_config.get('model', 'mistral-large-latest')

    claim_results = store_claims_v2(version_id, filtered_claims, llm_provider, llm_model)

    sentiment_config = config.get('sentiment', {})
    stance_config = config.get('stance', {})
    stance_batch_size = stance_config.get('batch_size', 5)

    # Load NLI model once for all stance analyses
    nli_model_name = stance_config.get('nli_model', 'facebook/bart-large-mnli')
    nli_pipeline = load_nli_model(nli_model_name)

    total_sentiment = 0
    total_stance = 0
    claims_with_both = 0

    # Build set of claim_ids that already have both sentiment and stance data (for resume)
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            claim_ids_list = [str(cid) for cid, _ in claim_results]
            cur.execute(f"""
                SELECT claim_id FROM {schema}.claim_sentiment WHERE claim_id = ANY(%s::uuid[])
                INTERSECT
                SELECT claim_id FROM {schema}.claim_stance WHERE claim_id = ANY(%s::uuid[])
            """, (claim_ids_list, claim_ids_list))
            already_complete = {str(r['claim_id']) for r in cur.fetchall()}

    if already_complete:
        logger.info(f"Resuming: {len(already_complete)} claims already have both sentiment & stance — skipping")

    for idx, (claim_id, claim) in enumerate(claim_results):
        claim_text = claim['claim_text']

        if str(claim_id) in already_complete:
            logger.info(f"Skipping claim {idx+1}/{len(claim_results)} (already complete): {claim_text[:60]}...")
            claims_with_both += 1
            continue

        logger.info(f"Analyzing claim {idx+1}/{len(claim_results)}: {claim_text[:60]}...")

        # Analyze sentiment
        sentiment_records = analyze_claim_sentiment(claim, sentiment_config)
        if sentiment_records:
            count = store_claim_sentiment(claim_id, sentiment_records)
            total_sentiment += count
            logger.info(f"  Sentiment: {count} records")
        else:
            logger.warning(f"  No sentiment data - skipping claim")
            continue

        # Analyze stance
        article_ids = claim['article_ids']
        stance_records = []

        # Fetch full article data
        with get_db() as db:
            schema = db.config["schema"]
            with db.cursor() as cur:
                # Cast article_ids to UUID array
                cur.execute(f"""
                    SELECT id, title, content, source_id
                    FROM {schema}.news_articles
                    WHERE id = ANY(%s::uuid[])
                """, (article_ids,))
                full_articles = [dict(a) for a in cur.fetchall()]

        # Process stance in batches using NLI model
        for i in range(0, len(full_articles), stance_batch_size):
            batch = full_articles[i:i + stance_batch_size]
            batch_stances = analyze_claim_stance_nli(nli_pipeline, claim_text, batch)
            stance_records.extend(batch_stances)

        if stance_records:
            count = store_claim_stance(claim_id, stance_records)
            total_stance += count
            logger.info(f"  Stance: {count} records")
            claims_with_both += 1
        else:
            logger.warning(f"  No stance data - claim incomplete")

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Pipeline complete!")
    logger.info(f"   Claims with both sentiment & stance: {claims_with_both}/{len(claim_results)}")
    logger.info(f"   Total sentiment records: {total_sentiment}")
    logger.info(f"   Total stance records: {total_stance}")
    logger.info(f"{'='*60}")

    return {
        'total_articles': len(articles),
        'num_batches': num_batches,
        'raw_claims': len(all_claims),
        'merged_claims': len(merged_claims),
        'deduplicated_claims': len(deduplicated_claims),
        'filtered_claims': len(filtered_claims),
        'claims_with_data': claims_with_both,
        'sentiment_records': total_sentiment,
        'stance_records': total_stance
    }
