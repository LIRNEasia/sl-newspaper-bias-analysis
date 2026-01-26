"""Named Entity Recognition for news articles."""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Install with: pip install spacy")

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

from .db import get_db, load_config


class NERExtractor:
    """Extract named entities from text using spaCy or GLiNER."""

    def __init__(
        self,
        provider: str = "spacy",
        model: str = "en_core_web_trf",
        entity_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        random_seed: int = 42
    ):
        self.provider = provider
        self.model_name = model
        self.entity_types = entity_types or []
        self.confidence_threshold = confidence_threshold

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Load model
        print(f"Loading {provider} NER model: {model}...")
        if provider == "spacy":
            if not SPACY_AVAILABLE:
                raise ImportError("spaCy not installed")
            self.nlp = spacy.load(model)
            print(f"✓ Loaded spaCy model")
        elif provider == "gliner":
            if not GLINER_AVAILABLE:
                raise ImportError("GLiNER not installed")
            self.model = GLiNER.from_pretrained(model)
            print(f"✓ Loaded GLiNER model")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def extract_entities(self, text: str, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.

        Args:
            text: Text to extract entities from
            doc_id: Optional document ID for tracking

        Returns:
            List of entity dictionaries with keys:
                - entity_text: The entity text
                - entity_type: Entity type (PERSON, ORG, etc.)
                - start_char: Start position
                - end_char: End position
                - confidence: Confidence score (0-1)
                - context: Sentence containing the entity
        """
        if self.provider == "spacy":
            return self._extract_spacy(text, doc_id)
        elif self.provider == "gliner":
            return self._extract_gliner(text, doc_id)

    def _extract_spacy(self, text: str, doc_id: str = None) -> List[Dict[str, Any]]:
        """Extract entities using spaCy."""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # Filter by entity type if specified
            if self.entity_types and ent.label_ not in self.entity_types:
                continue

            # Get sentence context
            context = ent.sent.text if ent.sent else ""

            entities.append({
                "entity_text": ent.text,
                "entity_type": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "confidence": 1.0,  # spaCy doesn't provide confidence scores
                "context": context[:500]  # Limit context length
            })

        return entities

    def _extract_gliner(self, text: str, doc_id: str = None) -> List[Dict[str, Any]]:
        """Extract entities using GLiNER."""
        # GLiNER requires entity types to be specified
        labels = self.entity_types if self.entity_types else [
            "person", "organization", "location", "date", "event"
        ]

        predictions = self.model.predict_entities(text, labels)

        entities = []
        for pred in predictions:
            # Filter by confidence
            if pred.get("score", 0) < self.confidence_threshold:
                continue

            # Get sentence context (simplified)
            start = max(0, pred["start"] - 100)
            end = min(len(text), pred["end"] + 100)
            context = text[start:end]

            entities.append({
                "entity_text": pred["text"],
                "entity_type": pred["label"].upper(),
                "start_char": pred["start"],
                "end_char": pred["end"],
                "confidence": pred.get("score", 1.0),
                "context": context[:500]
            })

        return entities


def extract_entities_from_articles(
    result_version_id: str,
    ner_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Extract named entities from all articles for a specific version.

    Args:
        result_version_id: UUID of the result version
        ner_config: NER configuration (from version config)
        batch_size: Number of articles to process in memory at once

    Returns:
        Summary of extraction results
    """
    # Load config
    if ner_config is None:
        config = load_config()
        ner_config = config.get("ner", {})

    # Initialize extractor
    extractor = NERExtractor(
        provider=ner_config.get("provider", "spacy"),
        model=ner_config.get("model", "en_core_web_trf"),
        entity_types=ner_config.get("entity_types"),
        confidence_threshold=ner_config.get("confidence_threshold", 0.5),
        random_seed=ner_config.get("random_seed", 42)
    )

    # Load articles
    print(f"Loading articles for version {result_version_id}...")
    with get_db() as db:
        articles = db.get_articles()  # Get all articles from news_articles table

    print(f"Loaded {len(articles)} articles")

    # Process articles
    all_entities = []
    entity_counts = {"total": 0, "by_type": {}}

    for article in tqdm(articles, desc="Extracting entities"):
        # Combine title and content
        text = f"{article['title']}\n\n{article['content']}"

        # Extract entities
        entities = extractor.extract_entities(text, doc_id=str(article['id']))

        # Add article_id to each entity
        for ent in entities:
            ent["article_id"] = str(article['id'])
            all_entities.append(ent)

            # Update counts
            entity_counts["total"] += 1
            ent_type = ent["entity_type"]
            entity_counts["by_type"][ent_type] = entity_counts["by_type"].get(ent_type, 0) + 1

    # Store in database
    print("Saving entities to database...")
    with get_db() as db:
        db.store_named_entities(all_entities, result_version_id)

        # Compute and store aggregated statistics
        db.compute_entity_statistics(result_version_id)

    # Summary
    summary = {
        "total_articles": len(articles),
        "total_entities": entity_counts["total"],
        "entities_by_type": entity_counts["by_type"],
        "avg_entities_per_article": entity_counts["total"] / len(articles) if articles else 0
    }

    print("\nEntity Extraction Complete:")
    print(f"  Total articles: {summary['total_articles']}")
    print(f"  Total entities: {summary['total_entities']}")
    print(f"  Avg entities/article: {summary['avg_entities_per_article']:.1f}")
    print(f"\nEntities by type:")
    for ent_type, count in sorted(summary['entities_by_type'].items(),
                                   key=lambda x: x[1], reverse=True):
        print(f"    {ent_type}: {count:,}")

    return summary
