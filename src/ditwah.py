"""Ditwah Hurricane Hypothesis Stance Analysis."""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from src.db import Database
from src.llm import get_llm, BaseLLM
from src.prompts import load_prompt


def filter_ditwah_articles(clustering_version_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Filter articles related to Cyclone Ditwah.

    Args:
        clustering_version_id: Optional version ID to filter by event clusters.
                              If None, filters by keyword search.

    Returns:
        List of article dictionaries with id, title, content, source_id, date_posted
    """
    with Database() as db:
        schema = db.config["schema"]

        if clustering_version_id:
            # Filter by event clusters containing "ditwah" in cluster name/description
            with db.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT DISTINCT
                        n.id, n.title, n.content, n.source_id, n.date_posted, n.url
                    FROM {schema}.news_articles n
                    JOIN {schema}.article_clusters ac ON n.id = ac.article_id
                    JOIN {schema}.event_clusters ec ON ac.cluster_id = ec.id
                    WHERE ac.result_version_id = %s
                      AND (
                          LOWER(ec.cluster_name) LIKE %s OR
                          LOWER(ec.cluster_description) LIKE %s OR
                          LOWER(n.title) LIKE %s OR
                          LOWER(n.content) LIKE %s
                      )
                    ORDER BY n.date_posted DESC
                    """,
                    (clustering_version_id, '%ditwah%', '%ditwah%', '%ditwah%', '%ditwah%')
                )
                results = cur.fetchall()
        else:
            # Fallback: keyword search in title and content
            with db.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, title, content, source_id, date_posted, url
                    FROM {schema}.news_articles
                    WHERE LOWER(title) LIKE %s OR LOWER(content) LIKE %s
                    ORDER BY date_posted DESC
                    """,
                    ('%ditwah%', '%ditwah%')
                )
                results = cur.fetchall()

        return [
            {
                "id": str(row["id"]),
                "title": row["title"],
                "content": row["content"],
                "source_id": row["source_id"],
                "date_posted": row["date_posted"],
                "url": row.get("url", "")
            }
            for row in results
        ]


def generate_stance_prompt(article_title: str, article_content: str, hypothesis: str) -> str:
    """
    Generate prompt for LLM to analyze article stance towards hypothesis.

    Args:
        article_title: Article headline
        article_content: Article full text
        hypothesis: Hypothesis statement to evaluate

    Returns:
        Formatted prompt string
    """
    # Truncate content if too long (keep first 3000 chars)
    content_truncated = article_content[:3000] if len(article_content) > 3000 else article_content

    system_prompt = load_prompt("ditwah/stance_system.md")
    user_prompt = load_prompt(
        "ditwah/stance_user.md",
        article_title=article_title,
        article_content=content_truncated,
        hypothesis=hypothesis,
    )

    return system_prompt, user_prompt


def parse_stance_response(response_text: str) -> Dict[str, Any]:
    """
    Parse and validate LLM response for stance analysis.

    Args:
        response_text: Raw LLM response (should be JSON)

    Returns:
        Validated dictionary with stance analysis fields

    Raises:
        ValueError: If response is invalid
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON if wrapped in markdown
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            raise ValueError("Failed to parse JSON response")

    # Validate required fields
    required_fields = ["agreement_score", "confidence", "stance", "reasoning"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate ranges
    if not -1.0 <= data["agreement_score"] <= 1.0:
        raise ValueError(f"agreement_score must be between -1.0 and 1.0, got {data['agreement_score']}")

    if not 0.0 <= data["confidence"] <= 1.0:
        raise ValueError(f"confidence must be between 0.0 and 1.0, got {data['confidence']}")

    valid_stances = ["strongly_agree", "agree", "neutral", "disagree", "strongly_disagree"]
    if data["stance"] not in valid_stances:
        raise ValueError(f"stance must be one of {valid_stances}, got {data['stance']}")

    # Ensure supporting_quotes is a list
    if "supporting_quotes" not in data:
        data["supporting_quotes"] = []
    elif not isinstance(data["supporting_quotes"], list):
        data["supporting_quotes"] = [data["supporting_quotes"]]

    return data


def store_hypotheses(version_id: str, hypotheses: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Store hypotheses in database for a given version.

    Args:
        version_id: Result version UUID
        hypotheses: List of hypothesis dicts with keys: key, statement, category

    Returns:
        Dictionary mapping hypothesis_key to hypothesis_id (UUID)
    """
    with Database() as db:
        schema = db.config["schema"]
        hypothesis_map = {}

        for hyp in hypotheses:
            with db.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {schema}.ditwah_hypotheses
                        (result_version_id, hypothesis_key, statement, category)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (result_version_id, hypothesis_key)
                    DO UPDATE SET
                        statement = EXCLUDED.statement,
                        category = EXCLUDED.category
                    RETURNING id
                    """,
                    (version_id, hyp["key"], hyp["statement"], hyp.get("category"))
                )
                hypothesis_id = str(cur.fetchone()["id"])
                hypothesis_map[hyp["key"]] = hypothesis_id

        return hypothesis_map


def store_analysis_result(
    version_id: str,
    article_id: str,
    hypothesis_id: str,
    analysis: Dict[str, Any],
    llm_provider: str,
    llm_model: str,
    processing_time_ms: int
) -> None:
    """
    Store a single article-hypothesis analysis result.

    Args:
        version_id: Result version UUID
        article_id: Article UUID
        hypothesis_id: Hypothesis UUID
        analysis: Parsed analysis result from LLM
        llm_provider: LLM provider name
        llm_model: LLM model name
        processing_time_ms: Time taken for analysis
    """
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {schema}.ditwah_analyses
                    (result_version_id, article_id, hypothesis_id,
                     agreement_score, confidence, stance, reasoning,
                     supporting_quotes, llm_provider, llm_model, processing_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_id, hypothesis_id, result_version_id)
                DO UPDATE SET
                    agreement_score = EXCLUDED.agreement_score,
                    confidence = EXCLUDED.confidence,
                    stance = EXCLUDED.stance,
                    reasoning = EXCLUDED.reasoning,
                    supporting_quotes = EXCLUDED.supporting_quotes,
                    llm_provider = EXCLUDED.llm_provider,
                    llm_model = EXCLUDED.llm_model,
                    processed_at = NOW(),
                    processing_time_ms = EXCLUDED.processing_time_ms
                """,
                (
                    version_id, article_id, hypothesis_id,
                    analysis["agreement_score"], analysis["confidence"],
                    analysis["stance"], analysis["reasoning"],
                    json.dumps(analysis["supporting_quotes"]),
                    llm_provider, llm_model, processing_time_ms
                )
            )


def analyze_article_hypothesis(
    llm: BaseLLM,
    article: Dict[str, Any],
    hypothesis: Dict[str, str]
) -> Tuple[Dict[str, Any], int]:
    """
    Analyze a single article-hypothesis pair.

    Args:
        llm: LLM client instance
        article: Article dictionary
        hypothesis: Hypothesis dictionary

    Returns:
        Tuple of (analysis_result, processing_time_ms)
    """
    start_time = time.time()

    system_prompt, user_prompt = generate_stance_prompt(
        article["title"],
        article["content"],
        hypothesis["statement"]
    )

    try:
        response = llm.generate(user_prompt, system_prompt=system_prompt, json_mode=True)
        analysis = parse_stance_response(response.content)

        processing_time_ms = int((time.time() - start_time) * 1000)
        return analysis, processing_time_ms

    except Exception as e:
        print(f"  ⚠️  Error analyzing article: {e}")
        # Return neutral stance on error
        processing_time_ms = int((time.time() - start_time) * 1000)
        return {
            "agreement_score": 0.0,
            "confidence": 0.0,
            "stance": "neutral",
            "reasoning": f"Error during analysis: {str(e)}",
            "supporting_quotes": []
        }, processing_time_ms


def analyze_ditwah_stance(
    version_id: str,
    ditwah_config: Dict[str, Any],
    clustering_version_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main pipeline: Analyze Ditwah articles for stance towards hypotheses.

    Args:
        version_id: Result version UUID for this analysis
        ditwah_config: Configuration dictionary from version config
        clustering_version_id: Optional clustering version to filter articles

    Returns:
        Summary dictionary with statistics
    """
    print(f"\n{'='*60}")
    print("DITWAH HURRICANE STANCE ANALYSIS")
    print(f"{'='*60}\n")

    # 1. Filter Ditwah articles
    print("Step 1: Filtering Ditwah articles...")
    articles = filter_ditwah_articles(clustering_version_id)
    print(f"  Found {len(articles)} articles related to Cyclone Ditwah\n")

    if not articles:
        print("⚠️  No articles found. Exiting.")
        return {"articles_processed": 0, "hypotheses_count": 0}

    # 2. Store hypotheses
    print("Step 2: Storing hypotheses...")
    hypotheses = ditwah_config.get("hypotheses", [])
    hypothesis_map = store_hypotheses(version_id, hypotheses)
    print(f"  Stored {len(hypotheses)} hypotheses\n")

    # 3. Initialize LLM
    print("Step 3: Initializing LLM...")
    llm_config = ditwah_config.get("llm", {})

    if llm_config.get("provider") == "local":
        from src.llm import LocalLLM
        llm = LocalLLM(
            model=llm_config.get("model", "llama3.1:70b"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 1000)
        )
    else:
        llm = get_llm(llm_config)

    print(f"  LLM: {llm.provider} / {llm.model}\n")

    # 4. Analyze articles
    print("Step 4: Analyzing articles...")
    total_analyses = len(articles) * len(hypotheses)
    analyses_completed = 0
    analyses_failed = 0

    batch_size = ditwah_config.get("batch_size", 5)

    for article in articles:
        print(f"\n  Article: {article['title'][:60]}...")
        print(f"  Source: {article['source_id']} | Date: {article['date_posted']}")

        for hyp in hypotheses:
            hypothesis_id = hypothesis_map[hyp["key"]]

            print(f"    → Testing hypothesis {hyp['key']}: {hyp['statement'][:50]}...")

            try:
                analysis, processing_time_ms = analyze_article_hypothesis(
                    llm, article, hyp
                )

                store_analysis_result(
                    version_id=version_id,
                    article_id=article["id"],
                    hypothesis_id=hypothesis_id,
                    analysis=analysis,
                    llm_provider=llm.provider,
                    llm_model=llm.model,
                    processing_time_ms=processing_time_ms
                )

                analyses_completed += 1
                print(f"      ✓ Stance: {analysis['stance']} (score: {analysis['agreement_score']:.2f}, confidence: {analysis['confidence']:.2f})")

            except Exception as e:
                analyses_failed += 1
                print(f"      ✗ Failed: {e}")

        # Progress update
        progress = (analyses_completed + analyses_failed) / total_analyses * 100
        print(f"\n  Progress: {analyses_completed}/{total_analyses} ({progress:.1f}%)")

    # 5. Refresh materialized view
    print("\n\nStep 5: Refreshing aggregated view...")
    with Database() as db:
        schema = db.config["schema"]
        with db.cursor() as cur:
            cur.execute(f"REFRESH MATERIALIZED VIEW {schema}.ditwah_by_source")
    print("  ✓ View refreshed\n")

    # Summary
    summary = {
        "articles_processed": len(articles),
        "hypotheses_count": len(hypotheses),
        "total_analyses": total_analyses,
        "analyses_completed": analyses_completed,
        "analyses_failed": analyses_failed,
        "success_rate": analyses_completed / total_analyses * 100 if total_analyses > 0 else 0
    }

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Articles processed: {summary['articles_processed']}")
    print(f"Hypotheses tested: {summary['hypotheses_count']}")
    print(f"Total analyses: {summary['total_analyses']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"{'='*60}\n")

    return summary
