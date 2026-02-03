"""Sentiment analysis using local transformer model."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import numpy as np

from src.config import load_config


@dataclass
class SentimentResult:
    """Sentiment analysis result for an article."""
    article_id: str
    model_type: str  # 'local'
    model_name: str
    overall_sentiment: float  # -5 to +5
    overall_confidence: float
    headline_sentiment: float  # -5 to +5
    headline_confidence: float
    reasoning: Optional[str] = None
    aspects: Optional[Dict] = None
    processing_time_ms: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage."""
        return {
            "article_id": self.article_id,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "overall_sentiment": float(self.overall_sentiment),
            "overall_confidence": float(self.overall_confidence),
            "headline_sentiment": float(self.headline_sentiment),
            "headline_confidence": float(self.headline_confidence),
            "sentiment_reasoning": self.reasoning,
            "sentiment_aspects": self.aspects,
            "processing_time_ms": self.processing_time_ms
        }


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.sentiment_config = self.config["sentiment"]
        self.scale_min = self.sentiment_config["scale"]["min"]
        self.scale_max = self.sentiment_config["scale"]["max"]

    @abstractmethod
    def analyze_batch(
        self,
        articles: List[Dict],
        show_progress: bool = True
    ) -> List[SentimentResult]:
        """Analyze sentiment for a batch of articles.

        Args:
            articles: List of article dicts with 'id', 'title', 'content'
            show_progress: Whether to show progress bar

        Returns:
            List of SentimentResult objects
        """
        pass

    def _map_to_scale(self, score: float, from_range: tuple = (0, 1)) -> float:
        """Map a score from one range to the sentiment scale (-5 to +5).

        Args:
            score: Score to map
            from_range: Source range (min, max)

        Returns:
            Mapped score on sentiment scale
        """
        from_min, from_max = from_range
        # Normalize to 0-1
        normalized = (score - from_min) / (from_max - from_min)
        # Map to sentiment scale
        return self.scale_min + normalized * (self.scale_max - self.scale_min)

    def _map_from_negative_positive(
        self,
        negative: float,
        neutral: float,
        positive: float
    ) -> tuple[float, float]:
        """Map negative/neutral/positive probabilities to sentiment score and confidence.

        Args:
            negative: Probability of negative sentiment
            neutral: Probability of neutral sentiment
            positive: Probability of positive sentiment

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        # Calculate weighted sentiment
        # Negative contributes to negative end, positive to positive end
        sentiment_score = (positive - negative) * self.scale_max

        # Confidence is the maximum probability (how sure the model is)
        confidence = max(negative, neutral, positive)

        return sentiment_score, confidence


class LocalSentimentAnalyzer(SentimentAnalyzer):
    """Local transformer-based sentiment analyzer using cardiffnlp/twitter-roberta-base-sentiment."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_type = "roberta"  # Changed from "local" for consistency
        self.model_config = self.sentiment_config.get("roberta", self.sentiment_config.get("local_sentiment", {}))
        self.model_name = self.model_config["model"]
        self.batch_size = self.model_config["batch_size"]
        self.device = self.model_config.get("device", "cpu")

        # Lazy load model
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the transformer model and tokenizer."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            print(f"Loading sentiment model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f"Model loaded on {self.device}")

    def _analyze_text(self, text: str) -> tuple[float, float]:
        """Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        import torch

        self._load_model()

        # Tokenize and truncate to model's max length
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits[0]
            probabilities = torch.nn.functional.softmax(scores, dim=0)

        # Extract probabilities (negative, neutral, positive)
        probs = probabilities.cpu().numpy()
        negative, neutral, positive = probs[0], probs[1], probs[2]

        return self._map_from_negative_positive(negative, neutral, positive)

    def analyze_batch(
        self,
        articles: List[Dict],
        show_progress: bool = True
    ) -> List[SentimentResult]:
        """Analyze sentiment for a batch of articles."""
        results = []
        iterator = tqdm(articles, desc="Local sentiment analysis") if show_progress else articles

        for article in iterator:
            start_time = time.time()

            # Analyze title and content separately
            title = article.get("title", "")
            content = article.get("content", "")

            headline_sentiment, headline_confidence = self._analyze_text(title)
            overall_sentiment, overall_confidence = self._analyze_text(content)

            processing_time = int((time.time() - start_time) * 1000)

            result = SentimentResult(
                article_id=str(article["id"]),
                model_type=self.model_type,
                model_name=self.model_name,
                overall_sentiment=overall_sentiment,
                overall_confidence=overall_confidence,
                headline_sentiment=headline_sentiment,
                headline_confidence=headline_confidence,
                processing_time_ms=processing_time
            )
            results.append(result)

        return results


class DistilBERTSentimentAnalyzer(SentimentAnalyzer):
    """DistilBERT sentiment analyzer.

    Model: distilbert-base-uncased-finetuned-sst-2-english
    Output: 2 classes (LABEL_0=negative, LABEL_1=positive) - no neutral
    Mapping: (pos_prob - neg_prob) × 5 → [-5, +5]
    Confidence: max(neg_prob, pos_prob)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_type = "distilbert"
        self.model_config = self.sentiment_config.get("distilbert", {})
        self.model_name = self.model_config.get(
            "model",
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.batch_size = self.model_config.get("batch_size", 64)
        self.device = self.model_config.get("device", "cpu")
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._model is None:
            print(f"Loading sentiment model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f"Model loaded on {self.device}")

    def _analyze_text(self, text: str) -> tuple[float, float]:
        """Analyze single text. Returns (sentiment_score, confidence)."""
        if self._model is None:
            self._load_model()

        import torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        # LABEL_0=negative, LABEL_1=positive
        negative_prob = probs[0].item()
        positive_prob = probs[1].item()

        # Map to scale: (pos - neg) × scale_max
        sentiment_score = (positive_prob - negative_prob) * self.scale_max
        confidence = max(negative_prob, positive_prob)

        return sentiment_score, confidence

    def analyze_batch(self, articles: List[Dict], show_progress: bool = True):
        """Analyze batch of articles."""
        results = []

        iterator = articles
        if show_progress:
            iterator = tqdm(articles, desc=f"Analyzing with {self.model_type}")

        for article in iterator:
            start_time = time.time()

            headline_sentiment, headline_conf = self._analyze_text(article['title'])
            overall_sentiment, overall_conf = self._analyze_text(
                f"{article['title']}\n\n{article['content'][:8000]}"
            )

            processing_time = int((time.time() - start_time) * 1000)

            results.append(SentimentResult(
                article_id=str(article['id']),
                model_type=self.model_type,
                model_name=self.model_name,
                overall_sentiment=overall_sentiment,
                overall_confidence=overall_conf,
                headline_sentiment=headline_sentiment,
                headline_confidence=headline_conf,
                processing_time_ms=processing_time
            ))

        return results


class FinBERTSentimentAnalyzer(SentimentAnalyzer):
    """FinBERT sentiment analyzer optimized for financial news.

    Model: ProsusAI/finbert
    Output: 3 classes (negative, neutral, positive)
    Mapping: Same as RoBERTa - (pos - neg) × 5
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_type = "finbert"
        self.model_config = self.sentiment_config.get("finbert", {})
        self.model_name = self.model_config.get("model", "ProsusAI/finbert")
        self.batch_size = self.model_config.get("batch_size", 32)
        self.device = self.model_config.get("device", "cpu")
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if self._model is None:
            print(f"Loading sentiment model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            print(f"Model loaded on {self.device}")

    def _analyze_text(self, text: str) -> tuple[float, float]:
        """Same logic as RoBERTa - 3 classes."""
        if self._model is None:
            self._load_model()

        import torch
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        negative = probs[0].item()
        neutral = probs[1].item()
        positive = probs[2].item()

        return self._map_from_negative_positive(negative, neutral, positive)

    def analyze_batch(self, articles: List[Dict], show_progress: bool = True):
        """Analyze batch of articles."""
        results = []

        iterator = articles
        if show_progress:
            iterator = tqdm(articles, desc=f"Analyzing with {self.model_type}")

        for article in iterator:
            start_time = time.time()

            headline_sentiment, headline_conf = self._analyze_text(article['title'])
            overall_sentiment, overall_conf = self._analyze_text(
                f"{article['title']}\n\n{article['content'][:8000]}"
            )

            processing_time = int((time.time() - start_time) * 1000)

            results.append(SentimentResult(
                article_id=str(article['id']),
                model_type=self.model_type,
                model_name=self.model_name,
                overall_sentiment=overall_sentiment,
                overall_confidence=overall_conf,
                headline_sentiment=headline_sentiment,
                headline_confidence=headline_conf,
                processing_time_ms=processing_time
            ))

        return results


class VADERSentimentAnalyzer(SentimentAnalyzer):
    """VADER lexicon-based sentiment analyzer.

    Model: vaderSentiment (rule-based, no ML)
    Output: compound score in [-1, +1]
    Mapping: compound × 5 → [-5, +5]
    Advantage: Very fast, no GPU needed
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_type = "vader"
        self.model_name = "vaderSentiment"
        self.model_config = self.sentiment_config.get("vader", {})
        self.batch_size = self.model_config.get("batch_size", 128)
        self._analyzer = None

    def _load_model(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        if self._analyzer is None:
            print(f"Loading VADER sentiment analyzer...")
            self._analyzer = SentimentIntensityAnalyzer()
            print("VADER loaded")

    def _analyze_text(self, text: str) -> tuple[float, float]:
        """Analyze text with VADER."""
        if self._analyzer is None:
            self._load_model()

        scores = self._analyzer.polarity_scores(text)

        # compound score is in [-1, +1]
        compound = scores['compound']

        # Map to [-5, +5] scale
        sentiment_score = compound * self.scale_max

        # Confidence = absolute value of compound (stronger = more confident)
        confidence = abs(compound)

        return sentiment_score, confidence

    def analyze_batch(self, articles: List[Dict], show_progress: bool = True):
        """Analyze batch of articles."""
        results = []

        iterator = articles
        if show_progress:
            iterator = tqdm(articles, desc=f"Analyzing with {self.model_type}")

        for article in iterator:
            start_time = time.time()

            headline_sentiment, headline_conf = self._analyze_text(article['title'])
            overall_sentiment, overall_conf = self._analyze_text(
                f"{article['title']}\n\n{article['content'][:8000]}"
            )

            processing_time = int((time.time() - start_time) * 1000)

            results.append(SentimentResult(
                article_id=str(article['id']),
                model_type=self.model_type,
                model_name=self.model_name,
                overall_sentiment=overall_sentiment,
                overall_confidence=overall_conf,
                headline_sentiment=headline_sentiment,
                headline_confidence=headline_conf,
                processing_time_ms=processing_time
            ))

        return results


class TextBlobSentimentAnalyzer(SentimentAnalyzer):
    """TextBlob pattern-based sentiment analyzer.

    Model: TextBlob (pattern-based, rule-based)
    Output: polarity in [-1, +1], subjectivity in [0, 1]
    Mapping: polarity × 5 → [-5, +5]
    Confidence: 1 - subjectivity (objective = confident)
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.model_type = "textblob"
        self.model_name = "TextBlob"
        self.model_config = self.sentiment_config.get("textblob", {})
        self.batch_size = self.model_config.get("batch_size", 128)

    def _analyze_text(self, text: str) -> tuple[float, float]:
        """Analyze text with TextBlob."""
        from textblob import TextBlob

        blob = TextBlob(text)

        # polarity in [-1, +1]
        polarity = blob.sentiment.polarity

        # subjectivity in [0, 1] (0=objective, 1=subjective)
        # Use inverse as confidence
        confidence = 1.0 - blob.sentiment.subjectivity

        # Map polarity to [-5, +5] scale
        sentiment_score = polarity * self.scale_max

        return sentiment_score, confidence

    def analyze_batch(self, articles: List[Dict], show_progress: bool = True):
        """Analyze batch of articles."""
        results = []

        iterator = articles
        if show_progress:
            iterator = tqdm(articles, desc=f"Analyzing with {self.model_type}")

        for article in iterator:
            start_time = time.time()

            headline_sentiment, headline_conf = self._analyze_text(article['title'])
            overall_sentiment, overall_conf = self._analyze_text(
                f"{article['title']}\n\n{article['content'][:8000]}"
            )

            processing_time = int((time.time() - start_time) * 1000)

            results.append(SentimentResult(
                article_id=str(article['id']),
                model_type=self.model_type,
                model_name=self.model_name,
                overall_sentiment=overall_sentiment,
                overall_confidence=overall_conf,
                headline_sentiment=headline_sentiment,
                headline_confidence=headline_conf,
                processing_time_ms=processing_time
            ))

        return results


def get_sentiment_analyzer(model_type: str, config: dict = None) -> SentimentAnalyzer:
    """Factory function to get sentiment analyzer by type.

    Args:
        model_type: One of 'roberta', 'distilbert', 'finbert', 'vader', 'textblob'
        config: Optional config dict

    Returns:
        Appropriate SentimentAnalyzer instance
    """
    analyzers = {
        'roberta': LocalSentimentAnalyzer,
        'distilbert': DistilBERTSentimentAnalyzer,
        'finbert': FinBERTSentimentAnalyzer,
        'vader': VADERSentimentAnalyzer,
        'textblob': TextBlobSentimentAnalyzer
    }

    if model_type not in analyzers:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(analyzers.keys())}")

    return analyzers[model_type](config)


def analyze_sentiment(
    articles: List[Dict],
    config: dict = None,
    show_progress: bool = True
) -> List[SentimentResult]:
    """Main API for sentiment analysis using local transformer model.

    Args:
        articles: List of article dicts with 'id', 'title', 'content'
        config: Optional configuration dict
        show_progress: Whether to show progress bar

    Returns:
        List of SentimentResult objects
    """
    analyzer = LocalSentimentAnalyzer(config)
    return analyzer.analyze_batch(articles, show_progress)


def get_sentiment_stats(results: List[SentimentResult]) -> Dict[str, Any]:
    """Get statistics from sentiment analysis results.

    Args:
        results: List of SentimentResult objects

    Returns:
        Dict with statistics
    """
    if not results:
        return {}

    sentiments = [r.overall_sentiment for r in results]
    confidences = [r.overall_confidence for r in results]

    return {
        "total": len(results),
        "avg_sentiment": np.mean(sentiments),
        "std_sentiment": np.std(sentiments),
        "min_sentiment": np.min(sentiments),
        "max_sentiment": np.max(sentiments),
        "avg_confidence": np.mean(confidences),
        "negative_count": sum(1 for s in sentiments if s < -0.5),
        "neutral_count": sum(1 for s in sentiments if -0.5 <= s <= 0.5),
        "positive_count": sum(1 for s in sentiments if s > 0.5)
    }
