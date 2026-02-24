"""
Ditwah Claims Analysis Module

Automatically generates claims about Cyclone Ditwah from articles,
then analyzes sentiment and stance for each claim across sources.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from src.db import get_db
from src.prompts import load_prompt

logger = logging.getLogger(__name__)


def filter_ditwah_articles() -> List[Dict]:
    """Get all articles where is_ditwah_cyclone = 1."""
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


# ============================================================================
# NEW: Two-Step Claims Generation Functions
# ============================================================================

def generate_individual_claim_for_article(llm, article: Dict, config: Dict) -> Optional[str]:
    """
    Generate ONE specific, debatable claim for a single article that can be clustered effectively.

    Args:
        llm: LLM client instance
        article: Article dictionary with title, content, etc.
        config: Configuration dict

    Returns:
        Claim text string or None if generation fails
    """
    prompt = load_prompt(
        "ditwah/claims_individual.md",
        article_title=article['title'],
        article_date=article['date_posted'],
        article_source=article['source_id'],
        article_content=article['content'][:2000] if article['content'] else article['title'],
    )

    try:
        response = llm.generate(prompt=prompt, json_mode=True)
        claim_data = json.loads(response.content)
        claim = claim_data.get('claim', '').strip()

        if claim and len(claim) > 10:  # Sanity check
            return claim
        else:
            logger.warning(f"Generated claim too short for article {article['id']}: '{claim}'")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for article {article['id']}: {e}")
        logger.error(f"Response: {response.content[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error generating claim for article {article['id']}: {e}")
        return None


def generate_individual_claims_batch(
    llm,
    articles: List[Dict],
    config: Dict,
    llm_provider: str,
    llm_model: str
) -> List[Dict]:
    """
    Generate individual claims for a batch of articles.

    Args:
        llm: LLM client instance
        articles: List of article dictionaries
        config: Configuration dict
        llm_provider: LLM provider name
        llm_model: LLM model name

    Returns:
        List of dicts with keys: article_id, claim_text, llm_provider, llm_model
    """
    batch_size = config.get('batch_size', 5)
    results = []

    for i, article in enumerate(articles):
        logger.info(f"  Processing article {i+1}/{len(articles)}: {article['title'][:60]}...")

        claim = generate_individual_claim_for_article(llm, article, config)

        if claim:
            results.append({
                'article_id': str(article['id']),
                'claim_text': claim,
                'llm_provider': llm_provider,
                'llm_model': llm_model
            })

        # Small delay to avoid overwhelming local LLM
        if (i + 1) % batch_size == 0:
            import time
            time.sleep(0.5)

    logger.info(f"Generated {len(results)} individual claims from {len(articles)} articles")
    return results


def store_individual_claims(
    version_id: UUID,
    claims: List[Dict]
) -> List[UUID]:
    """
    Store individual claims to database.

    Args:
        version_id: Result version ID
        claims: List of dicts with keys: article_id, claim_text, llm_provider, llm_model

    Returns:
        List of individual claim IDs
    """
    with get_db() as db:
        schema = db.config["schema"]
        claim_ids = []

        with db.cursor() as cur:
            for claim in claims:
                cur.execute(f"""
                    INSERT INTO {schema}.ditwah_article_claims (
                        article_id,
                        result_version_id,
                        claim_text,
                        llm_provider,
                        llm_model
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, result_version_id) DO UPDATE
                    SET claim_text = EXCLUDED.claim_text,
                        llm_provider = EXCLUDED.llm_provider,
                        llm_model = EXCLUDED.llm_model
                    RETURNING id
                """, (
                    claim['article_id'],
                    str(version_id),
                    claim['claim_text'],
                    claim['llm_provider'],
                    claim['llm_model']
                ))
                claim_id = cur.fetchone()['id']
                claim_ids.append(claim_id)

        logger.info(f"Stored {len(claim_ids)} individual claims to database")
        return claim_ids


def cluster_individual_claims(
    version_id: UUID,
    config: Dict,
    max_clusters: int = 60,
    target_cluster_size: int = 35,
    min_articles: int = 20,
) -> List[List[str]]:
    """
    Cluster individual claims into groups using embeddings.

    n_clusters is computed dynamically from target_cluster_size so each cluster
    has approximately that many articles rather than using a fixed maximum.
    Clusters that fall below min_articles are merged into the nearest valid cluster
    so every article stays assigned to a claim.

    Args:
        version_id: Result version ID
        config: Configuration dict with clustering settings
        max_clusters: Hard upper bound on number of clusters (safety cap)
        target_cluster_size: Desired average articles per cluster (default 25)
        min_articles: Minimum articles a cluster must have; smaller ones are merged

    Returns:
        List of lists, where each inner list contains individual claim IDs
    """
    from src.llm import get_embeddings_client
    import numpy as np

    # Fetch individual claims
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT id, claim_text
                FROM {schema}.ditwah_article_claims
                WHERE result_version_id = %s
                ORDER BY created_at
            """, (str(version_id),))
            claims = cur.fetchall()

    if not claims:
        logger.warning("No individual claims found to cluster")
        return []

    total = len(claims)
    logger.info(f"Found {total} individual claims to cluster")

    # Generate embeddings for all individual claim texts
    embedding_config = config.get('embeddings', {})
    embeddings_client = get_embeddings_client(embedding_config)

    claim_texts = [c['claim_text'] for c in claims]
    claim_ids   = [str(c['id'])   for c in claims]

    logger.info("Generating embeddings for individual claims...")
    embeddings       = embeddings_client.embed(claim_texts)
    embeddings_array = np.array(embeddings)

    # -----------------------------------------------------------------------
    # Compute n_clusters dynamically based on target cluster size.
    # E.g. 1657 claims, target_size=25 → 1657//25 = 66 clusters
    # Hard-bounded by max_clusters from above.
    # -----------------------------------------------------------------------
    n_clusters = max(10, total // target_cluster_size)
    n_clusters = min(n_clusters, max_clusters, total)

    logger.info(
        f"Running KMeans clustering: {total} claims → {n_clusters} clusters "
        f"(target size ≈{target_cluster_size}, min ≥{min_articles})"
    )

    # Normalise embeddings → KMeans on unit vectors ≡ spherical k-means
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize as sk_normalize
    normed = sk_normalize(embeddings_array, norm='l2')
    rng = config.get('random_seed', 42)
    km = KMeans(n_clusters=n_clusters, random_state=rng, n_init=10, max_iter=300)
    labels = km.fit_predict(normed)

    # Map label → list of indices
    raw_clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        raw_clusters.setdefault(label, []).append(idx)

    # -----------------------------------------------------------------------
    # Merge clusters that are below min_articles into their nearest neighbour.
    # "Nearest" = smallest cosine distance between cluster centroids.
    # Cap: a target cluster may not grow beyond max_merge_size to prevent
    # one cluster from absorbing everything.
    # -----------------------------------------------------------------------
    max_merge_size = max(target_cluster_size * 3, min_articles * 4)

    def centroid(indices):
        return embeddings_array[indices].mean(axis=0)

    def cosine_dist(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    # Iteratively merge small clusters until all remaining meet min_articles
    changed = True
    while changed:
        changed = False
        small = [k for k, v in raw_clusters.items() if len(v) < min_articles]
        if not small:
            break

        # Pick the smallest cluster to merge first
        small.sort(key=lambda k: len(raw_clusters[k]))
        victim = small[0]
        victim_centroid = centroid(raw_clusters[victim])

        # Find candidate targets: other clusters that aren't over the size cap
        candidates = [
            k for k in raw_clusters
            if k != victim and len(raw_clusters[k]) <= max_merge_size
        ]
        # Fall back to all others if all are over cap (edge case)
        if not candidates:
            candidates = [k for k in raw_clusters if k != victim]
        if not candidates:
            break  # Only one cluster left

        nearest = min(
            candidates,
            key=lambda k: cosine_dist(victim_centroid, centroid(raw_clusters[k]))
        )

        # Merge victim into nearest
        raw_clusters[nearest].extend(raw_clusters.pop(victim))
        changed = True

    # Convert index lists → claim_id lists
    final_clusters = [
        [claim_ids[i] for i in indices]
        for indices in raw_clusters.values()
    ]

    sizes = sorted([len(c) for c in final_clusters], reverse=True)
    logger.info(
        f"Final: {len(final_clusters)} clusters | "
        f"min={min(sizes)} max={max(sizes)} avg={sum(sizes)/len(sizes):.1f}"
    )
    logger.info(f"Size distribution: {sizes}")

    return final_clusters


def generate_general_claim_from_cluster(
    llm,
    individual_claim_ids: List[str],
    version_id: UUID,
    config: Dict
) -> Optional[Dict]:
    """
    Generate a general claim from a cluster of individual claims.

    Fetches article titles and content excerpts alongside the per-article claims so
    the LLM has richer context when synthesising the general claim.

    Args:
        llm: LLM client instance
        individual_claim_ids: List of individual claim IDs in this cluster
        version_id: Result version ID
        config: Configuration dict

    Returns:
        Dict with keys: claim_text, claim_category or None if generation fails
    """
    # Fetch individual claims AND article context (title + excerpt)
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            placeholders = ','.join(['%s'] * len(individual_claim_ids))
            cur.execute(f"""
                SELECT ac.claim_text,
                       n.source_id,
                       n.title        AS article_title,
                       LEFT(n.content, 400) AS content_excerpt
                FROM {schema}.ditwah_article_claims ac
                JOIN {schema}.news_articles n ON ac.article_id = n.id
                WHERE ac.id IN ({placeholders})
            """, individual_claim_ids)
            claims_data = cur.fetchall()

    if not claims_data:
        logger.warning("No claims found for cluster")
        return None

    sources = list(set(c['source_id'] for c in claims_data))

    # All per-article claims — these are short (1-2 sentences) so include every one
    all_claims_block = '\n'.join(
        f"• [{row['source_id']}] {row['claim_text']}"
        for row in claims_data
    )

    # Article excerpts for richer context — show up to 20 articles with full text
    article_summaries = []
    for row in claims_data[:20]:
        excerpt = (row['content_excerpt'] or '').strip().replace('\n', ' ')
        article_summaries.append(
            f"• [{row['source_id']}] {row['article_title']}\n"
            f"  Excerpt: {excerpt[:300]}"
        )
    article_block = '\n\n'.join(article_summaries)
    extra = (
        f'\n\n(+ {len(claims_data) - 20} more articles not shown above)'
        if len(claims_data) > 20 else ''
    )

    categories = config.get('categories', [
        "government_response",
        "humanitarian_aid",
        "infrastructure_damage",
        "economic_impact",
        "international_response",
        "casualties_and_displacement",
        "relief_operations",
        "evacuation_and_displacement",
        "weather_warnings",
        "preparation_measures"
    ])

    individual_claims = [row['claim_text'] for row in claims_data]
    overflow_text = (
        f'... and {len(individual_claims) - 10} more' if len(individual_claims) > 10 else ''
    )
    prompt = load_prompt(
        "ditwah/claims_general.md",
        claim_count=len(individual_claims),
        sources_list=', '.join(sources),
        claims_list='\n'.join(
            f'{i+1}. {claim}' for i, claim in enumerate(individual_claims[:10])
        ),
        overflow_text=overflow_text,
        categories_list=', '.join(categories),
    )

    try:
        response = llm.generate(prompt=prompt, json_mode=True)
        result = json.loads(response.content)

        claim_text = result.get('claim_text', '').strip()
        claim_category = result.get('claim_category', 'other')

        if claim_text and len(claim_text) > 10:
            return {
                'claim_text': claim_text,
                'claim_category': claim_category
            }
        else:
            logger.warning(f"Generated general claim too short: '{claim_text}'")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON for general claim: {e}")
        logger.error(f"Response: {response.content[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error generating general claim: {e}")
        return None


def store_general_claims_and_link(
    version_id: UUID,
    clusters: List[List[str]],
    general_claims_data: List[Dict],
    llm_provider: str,
    llm_model: str
) -> List[UUID]:
    """
    Store general claims and link individual claims to them.

    Args:
        version_id: Result version ID
        clusters: List of lists of individual claim IDs
        general_claims_data: List of dicts with claim_text and claim_category
        llm_provider: LLM provider name
        llm_model: LLM model name

    Returns:
        List of general claim IDs
    """
    general_claim_ids = []

    with get_db() as db:
        schema = db.config["schema"]

        with db.cursor() as cur:
            for idx, (cluster, general_claim_data) in enumerate(zip(clusters, general_claims_data)):
                if not general_claim_data:
                    continue

                # Insert general claim
                cur.execute(f"""
                    INSERT INTO {schema}.ditwah_claims (
                        result_version_id,
                        claim_text,
                        claim_category,
                        claim_order,
                        individual_claims_count,
                        llm_provider,
                        llm_model
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (result_version_id, claim_text) DO UPDATE
                    SET claim_category = EXCLUDED.claim_category,
                        claim_order = EXCLUDED.claim_order,
                        individual_claims_count = EXCLUDED.individual_claims_count
                    RETURNING id
                """, (
                    str(version_id),
                    general_claim_data['claim_text'],
                    general_claim_data['claim_category'],
                    idx + 1,  # claim_order
                    len(cluster),  # individual_claims_count
                    llm_provider,
                    llm_model
                ))
                general_claim_id = cur.fetchone()['id']
                general_claim_ids.append(general_claim_id)

                # Link individual claims to this general claim
                placeholders = ','.join(['%s'] * len(cluster))
                cur.execute(f"""
                    UPDATE {schema}.ditwah_article_claims
                    SET general_claim_id = %s
                    WHERE id IN ({placeholders})
                """, [str(general_claim_id)] + cluster)

                logger.info(f"  Stored general claim {idx+1}: '{general_claim_data['claim_text'][:60]}...' ({len(cluster)} individual claims)")

    logger.info(f"Stored {len(general_claim_ids)} general claims")
    return general_claim_ids


def generate_claims_from_articles(llm, articles: List[Dict], config: Dict) -> List[Dict]:
    """
    Send articles to LLM in batches, ask it to identify key claims.

    Args:
        llm: LLM client instance
        articles: List of article dictionaries
        config: Configuration dict with claim generation settings

    Returns:
        List of claim dictionaries with keys: claim_text, claim_category, confidence
    """
    logger.info(f"Generating claims from {len(articles)} articles...")

    num_claims = config.get('num_claims', 22)
    max_articles = config.get('max_articles_for_generation', 200)
    categories = config.get('categories', [
        "government_response",
        "international_response",
        "infrastructure_damage",
        "casualties_and_displacement",
        "economic_impact",
        "relief_operations",
        "weather_warnings",
        "recovery"
    ])

    # Sample articles evenly across sources to stay within LLM context limits
    if len(articles) > max_articles:
        import random
        from collections import defaultdict
        by_source = defaultdict(list)
        for a in articles:
            by_source[a['source_id']].append(a)
        sampled = []
        per_source = max_articles // max(len(by_source), 1)
        for source_articles in by_source.values():
            random.shuffle(source_articles)
            sampled.extend(source_articles[:per_source])
        # Top up to max_articles if needed
        remaining = [a for a in articles if a not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[:max_articles - len(sampled)])
        articles_to_use = sampled[:max_articles]
        logger.info(f"Sampled {len(articles_to_use)} articles (from {len(articles)}) spread across {len(by_source)} sources")
    else:
        articles_to_use = articles

    # Prepare article summaries for LLM (title + first 400 chars)
    article_summaries = []
    for article in articles_to_use:
        summary = {
            'title': article['title'],
            'excerpt': article['content'][:400] if article['content'] else '',
            'source_id': article['source_id']
        }
        article_summaries.append(summary)

    # Create LLM prompt
    prompt = load_prompt(
        "ditwah/claims_batch.md",
        article_count=len(articles),
        num_claims=num_claims,
        articles_json=json.dumps(article_summaries, indent=2),
        categories_list=', '.join(categories),
    )

    try:
        # Call LLM
        response = llm.generate(
            prompt=prompt,
            json_mode=True
        )

        # Clean response content (remove markdown code blocks if present)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]  # Remove ```json
        if content.startswith('```'):
            content = content[3:]  # Remove ```
        if content.endswith('```'):
            content = content[:-3]  # Remove trailing ```
        content = content.strip()

        # Parse JSON response
        claims = json.loads(content)

        if not isinstance(claims, list):
            logger.error("LLM response is not a list")
            return []

        # Normalize key names — LLMs sometimes return "category" instead of "claim_category"
        for claim in claims:
            if 'category' in claim and 'claim_category' not in claim:
                claim['claim_category'] = claim.pop('category')
            if 'claim_category' not in claim:
                claim['claim_category'] = 'other'
            if 'claim_text' not in claim and 'text' in claim:
                claim['claim_text'] = claim.pop('text')

        logger.info(f"Generated {len(claims)} claims")
        return claims

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response: {response.content[:500]}...")
        return []
    except Exception as e:
        logger.error(f"Error generating claims: {e}")
        return []


def store_claims(version_id: UUID, claims: List[Dict], llm_provider: str, llm_model: str) -> List[UUID]:
    """
    Store generated claims in database.

    Returns:
        List of claim IDs
    """
    with get_db() as db:
        schema = db.config["schema"]
        claim_ids = []

        with db.cursor() as cur:
            for i, claim in enumerate(claims):
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
                    RETURNING id
                """, (
                    str(version_id),
                    claim['claim_text'],
                    claim['claim_category'],
                    i + 1,  # claim_order
                    llm_provider,
                    llm_model
                ))
                claim_id = cur.fetchone()['id']
                claim_ids.append(claim_id)

        logger.info(f"Stored {len(claim_ids)} claims")
        return claim_ids


def store_claim_sentiment(claim_id: UUID, sentiment_records: List[Dict]) -> int:
    """
    Store sentiment records to database with ON CONFLICT handling.

    Args:
        claim_id: UUID of the claim
        sentiment_records: List of dicts with keys: article_id, source_id, sentiment_score, sentiment_model

    Returns:
        Count of records stored
    """
    if not sentiment_records:
        return 0

    with get_db() as db:
        schema = db.config["schema"]
        count = 0

        with db.cursor() as cur:
            for record in sentiment_records:
                cur.execute(f"""
                    INSERT INTO {schema}.claim_sentiment (
                        claim_id,
                        article_id,
                        source_id,
                        sentiment_score,
                        sentiment_model
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (claim_id, article_id) DO UPDATE
                    SET sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_model = EXCLUDED.sentiment_model
                """, (
                    str(claim_id),
                    record['article_id'],
                    record['source_id'],
                    record['sentiment_score'],
                    record['sentiment_model']
                ))
                count += 1

        logger.info(f"Stored {count} sentiment records for claim {claim_id}")
        return count


def store_claim_stance(claim_id: UUID, stance_records: List[Dict]) -> int:
    """
    Store stance records to database with ON CONFLICT handling.

    Args:
        claim_id: UUID of the claim
        stance_records: List of dicts with keys: article_id, source_id, stance_score, stance_label,
                       confidence, reasoning, supporting_quotes, llm_provider, llm_model

    Returns:
        Count of records stored
    """
    if not stance_records:
        return 0

    with get_db() as db:
        schema = db.config["schema"]
        count = 0

        with db.cursor() as cur:
            for record in stance_records:
                cur.execute(f"""
                    INSERT INTO {schema}.claim_stance (
                        claim_id,
                        article_id,
                        source_id,
                        stance_score,
                        stance_label,
                        confidence,
                        reasoning,
                        supporting_quotes,
                        llm_provider,
                        llm_model
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (claim_id, article_id) DO UPDATE
                    SET stance_score = EXCLUDED.stance_score,
                        stance_label = EXCLUDED.stance_label,
                        confidence = EXCLUDED.confidence,
                        reasoning = EXCLUDED.reasoning,
                        supporting_quotes = EXCLUDED.supporting_quotes,
                        llm_provider = EXCLUDED.llm_provider,
                        llm_model = EXCLUDED.llm_model,
                        processed_at = NOW()
                """, (
                    str(claim_id),
                    record['article_id'],
                    record['source_id'],
                    record['stance_score'],
                    record['stance_label'],
                    record['confidence'],
                    record['reasoning'],
                    record['supporting_quotes'],  # Already JSON string
                    record['llm_provider'],
                    record['llm_model']
                ))
                count += 1

        logger.info(f"Stored {count} stance records for claim {claim_id}")
        return count


def identify_articles_mentioning_claim(claim_text: str, articles: List[Dict]) -> List[Dict]:
    """
    Identify which articles mention or relate to a claim using keyword matching.

    Args:
        claim_text: The claim text
        articles: List of all Ditwah articles

    Returns:
        List of articles that mention the claim
    """
    # Extract key terms from claim (simple approach - words > 4 chars, excluding common words)
    stop_words = {'the', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
    keywords = [
        word.lower()
        for word in claim_text.split()
        if len(word) > 4 and word.lower() not in stop_words
    ][:5]  # Top 5 keywords

    matching_articles = []
    for article in articles:
        content_lower = (article['title'] + ' ' + (article['content'] or '')).lower()

        # Check if at least 2 keywords appear in the article
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        if matches >= 2:
            matching_articles.append(article)

    logger.info(f"Found {len(matching_articles)} articles mentioning claim: '{claim_text[:60]}...'")
    return matching_articles


def analyze_claim_sentiment_to_df(
    claim_index: int,
    claim_text: str,
    articles: List[Dict],
    sentiment_model: str = 'roberta'
) -> List[Dict]:
    """
    For each article mentioning the claim, fetch existing sentiment score
    and return as list of dictionaries (for dataframe).

    Args:
        claim_index: Index of the claim (0-based)
        claim_text: Text of the claim
        articles: List of articles mentioning this claim
        sentiment_model: Which sentiment model to use (default: 'roberta')

    Returns:
        List of sentiment record dictionaries
    """
    with get_db() as db:
        schema = db.config["schema"]
        records = []

        with db.cursor() as cur:
            for article in articles:
                # Fetch existing sentiment score
                cur.execute(f"""
                    SELECT overall_sentiment, model_name
                    FROM {schema}.sentiment_analyses
                    WHERE article_id = %s AND model_type = %s
                    LIMIT 1
                """, (str(article['id']), sentiment_model))

                sentiment = cur.fetchone()
                if not sentiment:
                    logger.warning(f"No sentiment found for article {article['id']}")
                    continue

                # Add to records list
                records.append({
                    'claim_index': claim_index,
                    'claim_text': claim_text,
                    'article_id': str(article['id']),
                    'source_id': article['source_id'],
                    'sentiment_score': sentiment['overall_sentiment'],
                    'sentiment_model': sentiment['model_name']
                })

        logger.info(f"Collected sentiment for {len(records)} articles for claim: '{claim_text[:60]}...'")
        return records


def analyze_claim_sentiment(claim_id: UUID, articles: List[Dict], sentiment_model: str = 'roberta') -> int:
    """
    DEPRECATED: Use analyze_claim_sentiment_to_df() instead.

    For each article mentioning the claim, fetch existing sentiment score
    and store in claim_sentiment table.

    Args:
        claim_id: UUID of the claim
        articles: List of articles mentioning this claim
        sentiment_model: Which sentiment model to use (default: 'roberta')

    Returns:
        Number of sentiment records created
    """
    with get_db() as db:
        schema = db.config["schema"]
        count = 0

        with db.cursor() as cur:
            for article in articles:
                # Fetch existing sentiment score
                cur.execute(f"""
                    SELECT overall_sentiment, model_name
                    FROM {schema}.sentiment_analyses
                    WHERE article_id = %s AND model_type = %s
                    LIMIT 1
                """, (str(article['id']), sentiment_model))

                sentiment = cur.fetchone()
                if not sentiment:
                    logger.warning(f"No sentiment found for article {article['id']}")
                    continue

                # Store in claim_sentiment
                cur.execute(f"""
                    INSERT INTO {schema}.claim_sentiment (
                        claim_id,
                        article_id,
                        source_id,
                        sentiment_score,
                        sentiment_model
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (claim_id, article_id) DO UPDATE
                    SET sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_model = EXCLUDED.sentiment_model
                """, (
                    str(claim_id),
                    str(article['id']),
                    article['source_id'],
                    sentiment['overall_sentiment'],
                    sentiment['model_name']
                ))
                count += 1

        logger.info(f"Stored sentiment for {count} articles for claim {claim_id}")
        return count


def analyze_claim_stance_to_df(
    llm,
    claim_index: int,
    claim_text: str,
    articles: List[Dict],
    config: Dict,
    llm_provider: str,
    llm_model: str
) -> List[Dict]:
    """
    For each article, use LLM to determine if it agrees/disagrees with the claim.
    Returns list of dictionaries (for dataframe) instead of storing to database.

    Args:
        llm: LLM client instance
        claim_index: Index of the claim (0-based)
        claim_text: The claim text
        articles: List of articles to analyze
        config: Stance analysis configuration
        llm_provider: LLM provider name
        llm_model: LLM model name

    Returns:
        List of stance record dictionaries
    """
    logger.info(f"Analyzing stance for {len(articles)} articles on claim: '{claim_text[:60]}...'")

    batch_size = config.get('batch_size', 5)
    temperature = config.get('temperature', 0.0)

    records = []

    # Process in batches
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]

        # Prepare article data for LLM
        article_data = []
        for article in batch:
            article_data.append({
                'id': str(article['id']),
                'title': article['title'],
                'content': article['content'][:1000] if article['content'] else '',  # First 1000 chars
                'source_id': article['source_id']
            })

        # Create LLM prompt
        prompt = load_prompt(
            "ditwah/claims_stance.md",
            claim_text=claim_text,
            articles_json=json.dumps(article_data, indent=2),
        )

        try:
            # Call LLM
            response = llm.generate(
                prompt=prompt,
                json_mode=True
            )

            # Parse JSON response
            stance_results = json.loads(response.content)

            if not isinstance(stance_results, list):
                logger.error(f"LLM response is not a list for batch {i}")
                continue

            # Collect results
            for result in stance_results:
                article_id = result['article_id']
                article = next((a for a in batch if str(a['id']) == article_id), None)
                if not article:
                    logger.warning(f"Article {article_id} not found in batch")
                    continue

                records.append({
                    'claim_index': claim_index,
                    'claim_text': claim_text,
                    'article_id': article_id,
                    'source_id': article['source_id'],
                    'stance_score': result['stance_score'],
                    'stance_label': result['stance_label'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'supporting_quotes': json.dumps(result.get('supporting_quotes', [])),
                    'llm_provider': llm_provider,
                    'llm_model': llm_model
                })

            logger.info(f"Processed batch {i//batch_size + 1}/{(len(articles) + batch_size - 1)//batch_size}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for batch {i}: {e}")
            logger.error(f"Response: {response.content[:500]}...")
            continue
        except Exception as e:
            logger.error(f"Error analyzing stance for batch {i}: {e}")
            continue

    logger.info(f"Collected stance for {len(records)} articles for claim: '{claim_text[:60]}...'")
    return records


def analyze_claim_stance(
    llm,
    claim_id: UUID,
    claim_text: str,
    articles: List[Dict],
    config: Dict,
    llm_provider: str,
    llm_model: str
) -> int:
    """
    DEPRECATED: Use analyze_claim_stance_to_df() instead.

    For each article, use LLM to determine if it agrees/disagrees with the claim.

    Args:
        llm: LLM client instance
        claim_id: UUID of the claim
        claim_text: The claim text
        articles: List of articles to analyze
        config: Stance analysis configuration

    Returns:
        Number of stance records created
    """
    logger.info(f"Analyzing stance for {len(articles)} articles on claim: '{claim_text[:60]}...'")

    batch_size = config.get('batch_size', 5)
    temperature = config.get('temperature', 0.0)

    count = 0

    # Process in batches
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]

        # Prepare article data for LLM
        article_data = []
        for article in batch:
            article_data.append({
                'id': str(article['id']),
                'title': article['title'],
                'content': article['content'][:1000] if article['content'] else '',  # First 1000 chars
                'source_id': article['source_id']
            })

        # Create LLM prompt
        prompt = load_prompt(
            "ditwah/claims_stance.md",
            claim_text=claim_text,
            articles_json=json.dumps(article_data, indent=2),
        )

        try:
            # Call LLM
            response = llm.generate(
                prompt=prompt,
                json_mode=True
            )

            # Parse JSON response
            stance_results = json.loads(response.content)

            if not isinstance(stance_results, list):
                logger.error(f"LLM response is not a list for batch {i}")
                continue

            # Store results
            with get_db() as db:
                schema = db.config["schema"]
                with db.cursor() as cur:
                    for result in stance_results:
                        article_id = result['article_id']
                        article = next((a for a in batch if str(a['id']) == article_id), None)
                        if not article:
                            logger.warning(f"Article {article_id} not found in batch")
                            continue

                        cur.execute(f"""
                            INSERT INTO {schema}.claim_stance (
                                claim_id,
                                article_id,
                                source_id,
                                stance_score,
                                stance_label,
                                confidence,
                                reasoning,
                                supporting_quotes,
                                llm_provider,
                                llm_model
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (claim_id, article_id) DO UPDATE
                            SET stance_score = EXCLUDED.stance_score,
                                stance_label = EXCLUDED.stance_label,
                                confidence = EXCLUDED.confidence,
                                reasoning = EXCLUDED.reasoning,
                                supporting_quotes = EXCLUDED.supporting_quotes,
                                processed_at = NOW()
                        """, (
                            str(claim_id),
                            article_id,
                            article['source_id'],
                            result['stance_score'],
                            result['stance_label'],
                            result['confidence'],
                            result['reasoning'],
                            json.dumps(result.get('supporting_quotes', [])),
                            llm_provider,
                            llm_model
                        ))
                        count += 1


            logger.info(f"Processed batch {i//batch_size + 1}/{(len(articles) + batch_size - 1)//batch_size}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for batch {i}: {e}")
            logger.error(f"Response: {response.content[:500]}...")
            continue
        except Exception as e:
            logger.error(f"Error analyzing stance for batch {i}: {e}")
            continue

    logger.info(f"Stored stance for {count} articles for claim {claim_id}")
    return count


def analyze_claim_stance_nli(
    analyzer,
    claim_id: UUID,
    claim_text: str,
    articles: List[Dict],
) -> int:
    """
    NLI-based stance detection using a pre-loaded NLIStanceAnalyzer.

    Uses roberta-large-mnli with overlapping token chunks to handle articles
    longer than the model's 512-token limit.  No LLM calls are made.

    Args:
        analyzer:   Pre-instantiated NLIStanceAnalyzer (reused across claims).
        claim_id:   UUID of the general claim.
        claim_text: The claim text (used as NLI hypothesis).
        articles:   List of article dicts with keys: id, title, content, source_id.

    Returns:
        Number of stance records stored.
    """
    if not articles:
        return 0

    logger.info(
        f"NLI stance: {len(articles)} articles for claim '{claim_text[:60]}…'"
    )

    # Build premises: title + full content
    premises = [
        f"{a['title']}\n{(a['content'] or '').strip()}"
        for a in articles
    ]

    nli_results = analyzer.predict_batch(premises, claim_text)

    stance_records = []
    for article, result in zip(articles, nli_results):
        stance_records.append({
            "article_id": str(article["id"]),
            "source_id": article["source_id"],
            "stance_score": result["stance_score"],
            "stance_label": result["stance_label"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "supporting_quotes": "[]",   # NLI does not extract quotes
            "llm_provider": "local",
            "llm_model": analyzer.MODEL_NAME,
        })

    return store_claim_stance(claim_id, stance_records)


def update_claim_article_counts(version_id: UUID) -> None:
    """
    Update article_count for all general claims in a version.
    Counts distinct articles linked via individual claims.
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            # Update article_count based on individual claims
            cur.execute(f"""
                UPDATE {schema}.ditwah_claims gc
                SET article_count = (
                    SELECT COUNT(DISTINCT ac.article_id)
                    FROM {schema}.ditwah_article_claims ac
                    WHERE ac.general_claim_id = gc.id
                )
                WHERE gc.result_version_id = %s
            """, (str(version_id),))

            # Also update representative_article_id (pick most recent article)
            cur.execute(f"""
                UPDATE {schema}.ditwah_claims gc
                SET representative_article_id = (
                    SELECT ac.article_id
                    FROM {schema}.ditwah_article_claims ac
                    JOIN {schema}.news_articles n ON ac.article_id = n.id
                    WHERE ac.general_claim_id = gc.id
                    ORDER BY n.date_posted DESC
                    LIMIT 1
                )
                WHERE gc.result_version_id = %s
            """, (str(version_id),))

        logger.info("Updated article counts and representative articles for claims")


def get_articles_for_general_claim(claim_id: UUID) -> List[Dict]:
    """
    Get all articles linked to a general claim.

    Primary path: ditwah_article_claims (newer pipeline).
    Fallback: claim_sentiment (older pipeline that lacks ditwah_article_claims rows).

    Args:
        claim_id: General claim ID

    Returns:
        List of article dictionaries
    """
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT n.id, n.title, n.content, n.source_id, n.date_posted, n.url
                FROM {schema}.ditwah_article_claims ac
                JOIN {schema}.news_articles n ON ac.article_id = n.id
                WHERE ac.general_claim_id = %s
                ORDER BY n.date_posted DESC
            """, (str(claim_id),))
            rows = cur.fetchall()

        if rows:
            return rows

        # Fallback for versions that used claim_sentiment as article linkage
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT DISTINCT n.id, n.title, n.content, n.source_id, n.date_posted, n.url
                FROM {schema}.claim_sentiment cs
                JOIN {schema}.news_articles n ON cs.article_id = n.id
                WHERE cs.claim_id = %s
                ORDER BY n.date_posted DESC
            """, (str(claim_id),))
            return cur.fetchall()


def link_sentiment_to_general_claims(version_id: UUID, sentiment_model: str = 'roberta') -> int:
    """
    Link existing sentiment data to general claims via individual article claims.

    Args:
        version_id: Result version ID
        sentiment_model: Which sentiment model to use

    Returns:
        Number of sentiment records created
    """
    logger.info("Linking sentiment data to general claims...")

    with get_db() as db:
        schema = db.config["schema"]
        count = 0

        with db.cursor() as cur:
            # Get all general claims for this version
            cur.execute(f"""
                SELECT id FROM {schema}.ditwah_claims
                WHERE result_version_id = %s
            """, (str(version_id),))
            general_claims = cur.fetchall()

            for gc in general_claims:
                claim_id = gc['id']

                # Get articles for this general claim
                articles = get_articles_for_general_claim(claim_id)

                for article in articles:
                    # Fetch existing sentiment score
                    cur.execute(f"""
                        SELECT overall_sentiment, model_name
                        FROM {schema}.sentiment_analyses
                        WHERE article_id = %s AND model_type = %s
                        LIMIT 1
                    """, (str(article['id']), sentiment_model))

                    sentiment = cur.fetchone()
                    if not sentiment:
                        logger.warning(f"No sentiment found for article {article['id']}")
                        continue

                    # Store in claim_sentiment
                    cur.execute(f"""
                        INSERT INTO {schema}.claim_sentiment (
                            claim_id,
                            article_id,
                            source_id,
                            sentiment_score,
                            sentiment_model
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (claim_id, article_id) DO UPDATE
                        SET sentiment_score = EXCLUDED.sentiment_score,
                            sentiment_model = EXCLUDED.sentiment_model
                    """, (
                        str(claim_id),
                        str(article['id']),
                        article['source_id'],
                        sentiment['overall_sentiment'],
                        sentiment['model_name']
                    ))
                    count += 1

    logger.info(f"Linked {count} sentiment records to general claims")
    return count


def generate_claims_pipeline(version_id: UUID, config: Dict) -> Dict[str, Any]:
    """
    Main pipeline for Ditwah claims analysis.

    Steps:
    1. Filter Ditwah articles
    2. Generate claims with LLM
    3. Store claims to database
    4. For each claim:
       a. Identify which articles mention it
       b. Analyze sentiment (from existing data)
       c. Analyze stance (new LLM calls)
       d. Store sentiment to database
       e. Store stance to database
    5. Update article counts

    Args:
        version_id: Result version ID
        config: Configuration dict

    Returns:
        Summary dict with counts
    """
    from src.llm import get_llm

    logger.info(f"Starting Ditwah claims pipeline for version {version_id}")

    # 1. Filter Ditwah articles
    articles = filter_ditwah_articles()
    if not articles:
        logger.error("No Ditwah articles found. Run 01_mark_ditwah_articles.py first.")
        return {'error': 'No Ditwah articles found'}

    # 2. Generate claims with LLM
    llm_config = config.get('llm', {})
    llm = get_llm(llm_config)

    generation_config = config.get('generation', {})
    claims = generate_claims_from_articles(llm, articles, generation_config)

    if not claims:
        logger.error("No claims generated")
        return {'error': 'No claims generated'}

    # 3. Store claims to database
    llm_provider = llm_config.get('provider', 'mistral')
    llm_model = llm_config.get('model', 'mistral-large-latest')
    claim_ids = store_claims(version_id, claims, llm_provider, llm_model)

    logger.info(f"✅ Stored {len(claim_ids)} claims to database")

    # 4. For each claim, analyze sentiment and stance
    sentiment_config = config.get('sentiment', {})
    stance_config = config.get('stance', {})

    total_sentiment_records = 0
    total_stance_records = 0

    for idx, (claim, claim_id) in enumerate(zip(claims, claim_ids)):
        claim_text = claim['claim_text']
        logger.info(f"Processing claim {idx + 1}/{len(claims)}: {claim_text[:60]}...")

        # Identify articles mentioning this claim
        matching_articles = identify_articles_mentioning_claim(claim_text, articles)

        if not matching_articles:
            logger.warning(f"No articles found for claim: '{claim_text[:60]}...'")
            continue

        # Analyze sentiment (from existing data)
        sentiment_records = []
        with get_db() as db:
            schema = db.config["schema"]
            with db.cursor() as cur:
                for article in matching_articles:
                    # Fetch existing sentiment score
                    cur.execute(f"""
                        SELECT overall_sentiment, model_name
                        FROM {schema}.sentiment_analyses
                        WHERE article_id = %s AND model_type = %s
                        LIMIT 1
                    """, (str(article['id']), sentiment_config.get('primary_model', 'roberta')))

                    sentiment = cur.fetchone()
                    if not sentiment:
                        logger.warning(f"No sentiment found for article {article['id']}")
                        continue

                    sentiment_records.append({
                        'article_id': str(article['id']),
                        'source_id': article['source_id'],
                        'sentiment_score': sentiment['overall_sentiment'],
                        'sentiment_model': sentiment['model_name']
                    })

        # Store sentiment to database
        if sentiment_records:
            count = store_claim_sentiment(claim_id, sentiment_records)
            total_sentiment_records += count

        # Analyze stance (new LLM calls) - process in batches
        stance_records = []
        batch_size = stance_config.get('batch_size', 5)

        for i in range(0, len(matching_articles), batch_size):
            batch = matching_articles[i:i + batch_size]

            # Prepare article data for LLM
            article_data = []
            for article in batch:
                article_data.append({
                    'id': str(article['id']),
                    'title': article['title'],
                    'content': article['content'][:1000] if article['content'] else '',
                    'source_id': article['source_id']
                })

            # Create LLM prompt
            prompt = load_prompt(
                "ditwah/claims_stance.md",
                claim_text=claim_text,
                articles_json=json.dumps(article_data, indent=2),
            )

            try:
                # Call LLM
                response = llm.generate(
                    prompt=prompt,
                    json_mode=True
                )

                # Parse JSON response
                stance_results = json.loads(response.content)

                if not isinstance(stance_results, list):
                    logger.error(f"LLM response is not a list for batch {i}")
                    continue

                # Collect results
                for result in stance_results:
                    article_id = result['article_id']
                    article = next((a for a in batch if str(a['id']) == article_id), None)
                    if not article:
                        logger.warning(f"Article {article_id} not found in batch")
                        continue

                    stance_records.append({
                        'article_id': article_id,
                        'source_id': article['source_id'],
                        'stance_score': result['stance_score'],
                        'stance_label': result['stance_label'],
                        'confidence': result['confidence'],
                        'reasoning': result['reasoning'],
                        'supporting_quotes': json.dumps(result.get('supporting_quotes', [])),
                        'llm_provider': llm_provider,
                        'llm_model': llm_model
                    })

                logger.info(f"  Processed stance batch {i//batch_size + 1}/{(len(matching_articles) + batch_size - 1)//batch_size}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON for batch {i}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error analyzing stance for batch {i}: {e}")
                continue

        # Store stance to database
        if stance_records:
            count = store_claim_stance(claim_id, stance_records)
            total_stance_records += count

    # 5. Update article counts
    update_claim_article_counts(version_id)

    summary = {
        'claims_generated': len(claims),
        'articles_analyzed': len(articles),
        'sentiment_records': total_sentiment_records,
        'stance_records': total_stance_records
    }

    logger.info(f"✅ Pipeline complete: {summary}")
    return summary


def search_claims(version_id: UUID, keyword: Optional[str] = None) -> List[Dict]:
    """Search claims by keyword using SQL LIKE."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            if keyword:
                keyword_pattern = f"%{keyword.lower()}%"
                cur.execute(f"""
                    SELECT * FROM {schema}.ditwah_claims
                    WHERE result_version_id = %s
                      AND LOWER(claim_text) LIKE %s
                    ORDER BY claim_order, article_count DESC
                    LIMIT 50
                """, (str(version_id), keyword_pattern))
            else:
                cur.execute(f"""
                    SELECT * FROM {schema}.ditwah_claims
                    WHERE result_version_id = %s
                    ORDER BY claim_order, article_count DESC
                """, (str(version_id),))

            return cur.fetchall()


def get_claim_sentiment_by_source(claim_id: UUID) -> List[Dict]:
    """Get average sentiment by source for a claim."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.source_id,
                    AVG(cs.sentiment_score) as avg_sentiment,
                    STDDEV(cs.sentiment_score) as stddev_sentiment,
                    COUNT(*) as article_count
                FROM {schema}.claim_sentiment cs
                WHERE cs.claim_id = %s
                GROUP BY cs.source_id
                ORDER BY avg_sentiment DESC
            """, (str(claim_id),))
            return cur.fetchall()


def get_claim_stance_by_source(claim_id: UUID) -> List[Dict]:
    """Get average stance by source for a claim."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    cs.source_id,
                    AVG(cs.stance_score) as avg_stance,
                    STDDEV(cs.stance_score) as stddev_stance,
                    AVG(cs.confidence) as avg_confidence,
                    COUNT(*) as article_count
                FROM {schema}.claim_stance cs
                WHERE cs.claim_id = %s
                GROUP BY cs.source_id
                ORDER BY avg_stance DESC
            """, (str(claim_id),))
            return cur.fetchall()


def get_claim_stance_breakdown(claim_id: UUID) -> List[Dict]:
    """Get stance distribution (agree/neutral/disagree percentages) by source."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    source_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN stance_score > 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as agree_pct,
                    SUM(CASE WHEN stance_score BETWEEN -0.2 AND 0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as neutral_pct,
                    SUM(CASE WHEN stance_score < -0.2 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as disagree_pct
                FROM {schema}.claim_stance
                WHERE claim_id = %s
                GROUP BY source_id
            """, (str(claim_id),))
            return cur.fetchall()


def get_claim_articles(claim_id: UUID, limit: int = 10) -> List[Dict]:
    """Get sample articles for a claim with sentiment/stance scores."""
    with get_db() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"""
                SELECT
                    n.id,
                    n.title,
                    n.content,
                    n.date_posted,
                    n.url,
                    n.source_id,
                    cs_sentiment.sentiment_score,
                    cs_stance.stance_score,
                    cs_stance.stance_label,
                    cs_stance.supporting_quotes
                FROM {schema}.claim_sentiment cs_sentiment
                JOIN {schema}.claim_stance cs_stance
                    ON cs_sentiment.article_id = cs_stance.article_id
                    AND cs_sentiment.claim_id = cs_stance.claim_id
                JOIN {schema}.news_articles n ON n.id = cs_sentiment.article_id
                WHERE cs_sentiment.claim_id = %s
                ORDER BY n.date_posted DESC
                LIMIT %s
            """, (str(claim_id), limit))
            return cur.fetchall()
