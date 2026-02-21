"""Entity stance detection using zero-shot NLI.

Detects stance toward named entities in news articles by:
1. Chunking articles into sentence windows
2. Matching NER entities to chunks by character position
3. Scoring stance using NLI (entailment vs contradiction)
4. Filtering out neutral stances
"""

import time
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def chunk_by_sentences(content: str, chunk_size: int = 5) -> List[Dict[str, Any]]:
    """Split article content into overlapping sentence-window chunks.

    Uses spaCy sentencizer for lightweight sentence boundary detection.

    Args:
        content: Article text
        chunk_size: Number of sentences per chunk

    Returns:
        List of dicts with chunk_index, text, start_char, end_char
    """
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(content)

    sentences = list(doc.sents)
    if not sentences:
        return []

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk_sents = sentences[i:i + chunk_size]

        # Merge final chunk into previous if < 2 sentences
        if len(chunk_sents) < 2 and chunks:
            prev = chunks[-1]
            prev["text"] = content[prev["start_char"]:chunk_sents[-1].end_char]
            prev["end_char"] = chunk_sents[-1].end_char
            continue

        chunks.append({
            "chunk_index": len(chunks),
            "text": content[chunk_sents[0].start_char:chunk_sents[-1].end_char],
            "start_char": chunk_sents[0].start_char,
            "end_char": chunk_sents[-1].end_char
        })

    return chunks


def get_entities_in_chunk(
    entities: List[Dict[str, Any]],
    start_char: int,
    end_char: int,
    allowed_types: List[str]
) -> List[Dict[str, Any]]:
    """Filter entities that fall within a chunk's character range.

    Args:
        entities: List of entity dicts with start_char, end_char, entity_type, entity_text
        start_char: Chunk start position
        end_char: Chunk end position
        allowed_types: Entity types to include

    Returns:
        Deduplicated list of entities within the chunk
    """
    seen = set()
    result = []

    for e in entities:
        if e["entity_type"] not in allowed_types:
            continue
        # Entity overlaps with chunk
        if e["start_char"] >= start_char and e["start_char"] < end_char:
            key = e["entity_text"]
            if key not in seen:
                seen.add(key)
                result.append(e)

    return result


class NLIStanceScorer:
    """Score entity stance using NLI (Natural Language Inference)."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        print(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # NLI label mapping for this model: [contradiction, neutral, entailment]
        self.label_names = ["contradiction", "neutral", "entailment"]
        print("NLI model loaded.")

    def score_stances(
        self,
        chunk_text: str,
        entities: List[Dict[str, Any]],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """Score stance toward each entity in a chunk.

        For each entity, tests two hypotheses:
        - Positive: "This text portrays {entity} favorably."
        - Negative: "This text portrays {entity} unfavorably."

        Args:
            chunk_text: The text chunk
            entities: Entities found in this chunk
            batch_size: Batch size for NLI inference

        Returns:
            List of dicts with entity info + stance_score, confidence
        """
        if not entities:
            return []

        # Build all premise-hypothesis pairs
        pairs = []
        pair_meta = []  # Track which entity each pair belongs to

        for entity in entities:
            name = entity["entity_text"]
            pairs.append((chunk_text, f"This text portrays {name} favorably."))
            pairs.append((chunk_text, f"This text portrays {name} unfavorably."))
            pair_meta.append(("positive", entity))
            pair_meta.append(("negative", entity))

        # Batch inference
        all_probs = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self.tokenizer(
                premises, hypotheses,
                padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.extend(probs.tolist())

        # Process results: pair positive and negative for each entity
        results = []
        for i in range(0, len(all_probs), 2):
            pos_probs = all_probs[i]      # [contradiction, neutral, entailment]
            neg_probs = all_probs[i + 1]

            pos_entailment = pos_probs[2]  # P("favorable" is entailed)
            neg_entailment = neg_probs[2]  # P("unfavorable" is entailed)

            stance_score = pos_entailment - neg_entailment  # -1 to +1
            confidence = max(pos_entailment, neg_entailment)

            entity = pair_meta[i][1]
            results.append({
                "entity_text": entity["entity_text"],
                "entity_type": entity["entity_type"],
                "stance_score": round(stance_score, 4),
                "confidence": round(confidence, 4)
            })

        return results


def get_stance_label(score: float) -> str:
    """Map stance score to label.

    Args:
        score: Stance score from -1 to +1

    Returns:
        Label string
    """
    if score < -0.6:
        return "strongly_negative"
    elif score < -0.2:
        return "negative"
    elif score > 0.6:
        return "strongly_positive"
    elif score > 0.2:
        return "positive"
    else:
        return "neutral"


def entity_stance_pipeline(
    version_id: str,
    config: Dict[str, Any],
    limit: Optional[int] = None,
    batch_size: int = 16
) -> Dict[str, Any]:
    """Main pipeline: chunk articles, score entity stances, store results.

    Args:
        version_id: Entity stance version UUID
        config: Version configuration dict
        limit: Max articles to process (None for all)
        batch_size: NLI batch size

    Returns:
        Summary dict with processing statistics
    """
    from src.db import Database

    stance_config = config.get("entity_stance", {})
    ner_version_id = config.get("ner_version_id")

    if not ner_version_id:
        raise ValueError("ner_version_id must be set in configuration")

    chunk_size = stance_config.get("chunk_size", 5)
    neutral_threshold = stance_config.get("neutral_threshold", 0.2)
    min_confidence = stance_config.get("min_confidence", 0.3)
    entity_types = stance_config.get("entity_types", ["PERSON", "ORG", "GPE", "NORP", "EVENT", "LAW", "FAC"])
    model_name = stance_config.get("model", "cross-encoder/nli-deberta-v3-base")

    # Initialize NLI scorer
    scorer = NLIStanceScorer(model_name)

    start_time = time.time()

    with Database() as db:
        # Get articles that haven't been processed yet
        articles = db.get_articles_without_entity_stance(version_id, limit)
        total_articles = len(articles)
        print(f"Found {total_articles} articles to process")

        if total_articles == 0:
            return {"articles_processed": 0, "stances_stored": 0}

        total_stances = 0
        total_entities_scored = 0
        total_chunks = 0

        for idx, article in enumerate(articles):
            article_id = str(article["id"])
            content = article["content"]

            if not content or not content.strip():
                continue

            # Get NER entities for this article
            entities = db.get_entities_for_article(article_id, ner_version_id)
            if not entities:
                continue

            # Chunk the article
            chunks = chunk_by_sentences(content, chunk_size)
            total_chunks += len(chunks)

            article_stances = []

            for chunk in chunks:
                # Find entities in this chunk
                chunk_entities = get_entities_in_chunk(
                    entities, chunk["start_char"], chunk["end_char"], entity_types
                )

                if not chunk_entities:
                    continue

                # Score stances
                results = scorer.score_stances(chunk["text"], chunk_entities, batch_size)
                total_entities_scored += len(results)

                for r in results:
                    score = r["stance_score"]
                    conf = r["confidence"]

                    # Filter neutral and low-confidence
                    if abs(score) <= neutral_threshold or conf < min_confidence:
                        continue

                    article_stances.append({
                        "article_id": article_id,
                        "chunk_index": chunk["chunk_index"],
                        "start_char": chunk["start_char"],
                        "end_char": chunk["end_char"],
                        "entity_text": r["entity_text"],
                        "entity_type": r["entity_type"],
                        "stance_score": score,
                        "stance_label": get_stance_label(score),
                        "confidence": conf
                    })

            # Store stances for this article
            if article_stances:
                db.store_entity_stances(article_stances, version_id, ner_version_id)
                total_stances += len(article_stances)

            if (idx + 1) % 50 == 0 or idx == total_articles - 1:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                print(f"  [{idx + 1}/{total_articles}] {rate:.1f} articles/sec, "
                      f"{total_stances} stances stored so far")

    elapsed = time.time() - start_time
    summary = {
        "articles_processed": total_articles,
        "chunks_created": total_chunks,
        "entities_scored": total_entities_scored,
        "stances_stored": total_stances,
        "elapsed_seconds": round(elapsed, 1)
    }

    print(f"\nPipeline complete:")
    print(f"  Articles processed: {total_articles}")
    print(f"  Chunks created: {total_chunks}")
    print(f"  Entity-chunk pairs scored: {total_entities_scored}")
    print(f"  Non-neutral stances stored: {total_stances}")
    print(f"  Time: {elapsed:.1f}s")

    return summary
