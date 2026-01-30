"""Article summarization using extractive, abstractive, and LLM-based methods."""

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from tqdm import tqdm

# Resolve the project root (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.db import Database, load_config
from src.llm import get_llm


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize summarizer with configuration.

        Args:
            config: Summarization configuration dictionary
        """
        self.config = config
        self.method = config.get("method", "textrank")
        self.summary_length = config.get("summary_length", "medium")

    @abstractmethod
    def summarize(self, text: str) -> str:
        """
        Generate summary for the given text.

        Args:
            text: Article text to summarize

        Returns:
            Summary text
        """
        pass

    def get_target_length(self) -> Tuple[int, int]:
        """
        Get target sentence and word counts based on summary_length setting.

        Returns:
            Tuple of (target_sentences, target_words)
        """
        length = self.summary_length.lower()
        if length == "short":
            return (
                self.config.get("short_sentences", 3),
                self.config.get("short_words", 50)
            )
        elif length == "long":
            return (
                self.config.get("long_sentences", 8),
                self.config.get("long_words", 150)
            )
        else:  # medium
            return (
                self.config.get("medium_sentences", 5),
                self.config.get("medium_words", 100)
            )

    def count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())


class TextRankSummarizer(BaseSummarizer):
    """Extractive summarization using TextRank algorithm."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer as SumyTextRank
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        self.parser_class = PlaintextParser
        self.tokenizer = Tokenizer("english")
        self.stemmer = Stemmer("english")
        self.summarizer = SumyTextRank(self.stemmer)
        self.summarizer.stop_words = get_stop_words("english")

    def summarize(self, text: str) -> str:
        """Generate TextRank summary."""
        if not text or not text.strip():
            return ""

        target_sentences, _ = self.get_target_length()

        try:
            from io import StringIO
            parser = self.parser_class.from_string(text, self.tokenizer)

            summary_sentences = self.summarizer(parser.document, target_sentences)

            summary = " ".join(str(sentence) for sentence in summary_sentences)
            return summary.strip()

        except Exception as e:
            print(f"TextRank error: {e}")
            return ""


class LexRankSummarizer(BaseSummarizer):
    """Extractive summarization using LexRank algorithm."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.lex_rank import LexRankSummarizer as SumyLexRank
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        self.parser_class = PlaintextParser
        self.tokenizer = Tokenizer("english")
        self.stemmer = Stemmer("english")
        self.summarizer = SumyLexRank(self.stemmer)
        self.summarizer.stop_words = get_stop_words("english")

    def summarize(self, text: str) -> str:
        """Generate LexRank summary."""
        if not text or not text.strip():
            return ""

        target_sentences, _ = self.get_target_length()

        try:
            # Parse the text
            from io import StringIO
            parser = self.parser_class.from_string(text, self.tokenizer)

            # Generate summary
            summary_sentences = self.summarizer(parser.document, target_sentences)

            # Combine sentences
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            return summary.strip()

        except Exception as e:
            print(f"LexRank error: {e}")
            return ""


class TransformerSummarizer(BaseSummarizer):
    """Abstractive summarization using transformer models (BART, T5, Pegasus)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from transformers import pipeline

        model_map = {
            "bart": "facebook/bart-large-cnn",
            "t5": "t5-base",
            "pegasus": "google/pegasus-xsum",
            "led": "allenai/led-base-16384",
            "bigbird-pegasus": "google/bigbird-pegasus-large-arxiv",
            "longt5": "google/long-t5-tglobal-base",
        }

        model_max_tokens_map = {
            "bart": 1024,
            "t5": 512,
            "pegasus": 512,
            "led": 16384,
            "bigbird-pegasus": 4096,
            "longt5": 4096,
        }

        model_name = model_map.get(self.method, "facebook/bart-large-cnn")
        model_key = self.method if self.method in model_map else "bart"
        self.model_key = model_key
        self.chunk_long_articles = config.get("chunk_long_articles", True)

        try:
            # Force CPU processing to avoid CUDA tokenization errors with special characters
            # GPU can hit "index out of bounds" errors on edge cases in real-world text
            self.summarizer = pipeline("summarization", model=model_name, device=-1)
            # Get the model's actual max position embeddings to truncate properly
            # T5 models use relative position encodings and don't have max_position_embeddings
            self.model_max_tokens = getattr(
                self.summarizer.model.config, 'max_position_embeddings',
                getattr(self.summarizer.model.config, 'n_positions', model_max_tokens_map[model_key])
            )
            self.tokenizer = self.summarizer.tokenizer
            # Keep tokenizer length aligned with our internal truncation
            if hasattr(self.tokenizer, "model_max_length"):
                self.tokenizer.model_max_length = self.model_max_tokens
            print(f"Loaded {model_name} on CPU (max {self.model_max_tokens} tokens)")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize input text to prevent tokenization issues.

        Args:
            text: Raw input text

        Returns:
            Cleaned text safe for tokenization
        """
        if not text:
            return ""

        # Remove null bytes and other problematic characters
        text = text.replace('\x00', '')

        # Normalize whitespace
        text = ' '.join(text.split())

        # Ensure text has actual content
        if not text.strip():
            return ""

        return text

    def _truncate_to_token_limit(self, text: str, max_tokens: int = None) -> str:
        """
        Truncate text to fit within the model's token limit.

        Uses the model's own tokenizer for exact truncation instead of
        word-count heuristics.
        """
        if max_tokens is None:
            max_tokens = self.model_max_tokens

        # Leave room for special tokens (BOS/EOS)
        max_tokens = max_tokens - 2

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text

        # Truncate and decode back to text
        truncated_ids = token_ids[:max_tokens]
        return self.tokenizer.decode(truncated_ids, skip_special_tokens=True)

    def _prepare_inputs_with_kwargs(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare inputs and model-specific generation kwargs.

        Returns:
            Tuple of (text, generation_kwargs)
        """
        kwargs: Dict[str, Any] = {}

        if self.model_key == "bart":
            # BART's default max_length (142) is too restrictive and causes
            # mid-sentence truncation. Override it to allow complete sentences.
            kwargs["max_length"] = 300

        elif self.model_key == "led":
            # LED pipeline automatically handles global attention on the first token
            # Just pass text and let the pipeline handle tokenization
            pass

        elif self.model_key == "bigbird-pegasus":
            pass

        # Standard models (T5, Pegasus, LongT5) use their defaults

        return text, kwargs

    def summarize(self, text: str) -> str:
        """
        Generate transformer-based summary.

        Note: This method does not enforce hard token limits. The model
        generates summaries using its natural stopping behavior (EOS token).
        Target lengths in config are informational only for transformers.
        """
        if not text or not text.strip():
            return ""

        # Sanitize input text
        text = self._sanitize_text(text)
        if not text:
            return ""

        # Skip articles that are too short (likely to cause tokenization issues)
        word_count = self.count_words(text)
        if word_count < 10:
            return ""

        _, target_words = self.get_target_length()

        try:
            # Check token count to decide whether to chunk
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_count = len(token_ids)

            if token_count > self.model_max_tokens and self.chunk_long_articles:
                # Chunk the article and summarize each chunk
                return self._summarize_long_article(text, target_words)
            else:
                # Truncate to model's token limit to prevent index errors
                text = self._truncate_to_token_limit(text)

                # Prepare inputs with model-specific parameters
                inputs, gen_kwargs = self._prepare_inputs_with_kwargs(text)

                # Generate summary using the pipeline
                summary = self.summarizer(
                    inputs,
                    do_sample=False,
                    truncation=True,  # Only truncate INPUT if needed
                    **gen_kwargs,
                )
                return summary[0]["summary_text"].strip()

        except Exception as e:
            # Print error with text preview to help debug
            text_preview = text[:100].replace('\n', ' ')
            print(f"Transformer summarization error: {e} (text starts: '{text_preview}...')")
            return ""

    def _summarize_long_article(self, text: str, target_words: int) -> str:
        """Summarize long articles by chunking."""
        # Note: text is already sanitized by the caller
        # Chunk by tokens, not words, to respect model limits
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_token_size = self.model_max_tokens - 2  # Room for special tokens

        # Split token IDs into chunks
        chunks = []
        for i in range(0, len(token_ids), chunk_token_size):
            chunk_ids = token_ids[i:i + chunk_token_size]
            chunk = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunk = self._sanitize_text(chunk)
            if chunk and len(chunk.split()) >= 10:
                chunks.append(chunk)

        if not chunks:
            return ""

        # Summarize each chunk
        chunk_summaries = []
        words_per_chunk = max(20, target_words // len(chunks))

        for chunk in chunks:
            try:
                # Prepare inputs with model-specific parameters
                inputs, gen_kwargs = self._prepare_inputs_with_kwargs(chunk)

                # Generate summary using the pipeline
                summary = self.summarizer(
                    inputs,
                    do_sample=False,
                    truncation=True,
                    **gen_kwargs,
                )
                chunk_summaries.append(summary[0]["summary_text"])
            except Exception as e:
                print(f"Chunk summarization error: {e}")
                continue

        # Combine chunk summaries
        return " ".join(chunk_summaries).strip()


class LLMSummarizer(BaseSummarizer):
    """LLM-based summarization using Claude or GPT."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        llm_model = config.get("llm_model", "claude-sonnet-4-20250514")
        llm_temperature = config.get("llm_temperature", 0.0)

        # Determine provider from model name
        if "claude" in llm_model.lower():
            provider = "claude"
        elif "gpt" in llm_model.lower():
            provider = "openai"
        else:
            provider = "claude"  # Default

        # Create LLM config
        llm_config = {
            "provider": provider,
            "model": llm_model,
            "temperature": llm_temperature,
            "max_tokens": 4096
        }

        self.llm = get_llm(llm_config)

    def _load_prompt_template(self) -> str:
        """Load the summarization prompt template from file."""
        prompt_path = os.path.join(_PROJECT_ROOT, "prompts", "summarization.md")
        with open(prompt_path, "r") as f:
            return f.read()

    def summarize(self, text: str) -> str:
        """Generate LLM-based summary."""
        if not text or not text.strip():
            return ""

        target_sentences, target_words = self.get_target_length()

        template = self._load_prompt_template()
        prompt = template.replace("{{target_sentences}}", str(target_sentences))
        prompt = prompt.replace("{{target_words}}", str(target_words))
        prompt = prompt.replace("{{article_text}}", text)

        try:
            response = self.llm.generate(prompt)
            return response.content.strip()

        except Exception as e:
            print(f"LLM summarization error: {e}")
            return ""


def get_summarizer(config: Dict[str, Any]) -> BaseSummarizer:
    """
    Factory function to get the appropriate summarizer based on config.

    Args:
        config: Summarization configuration

    Returns:
        Summarizer instance

    Raises:
        ValueError: If method is not supported
    """
    method = config.get("method", "textrank").lower()

    if method == "textrank":
        return TextRankSummarizer(config)
    elif method == "lexrank":
        return LexRankSummarizer(config)
    elif method in ["bart", "t5", "pegasus", "led", "bigbird-pegasus", "longt5"]:
        return TransformerSummarizer(config)
    elif method in ["claude", "gpt"]:
        return LLMSummarizer(config)
    else:
        raise ValueError(f"Unsupported summarization method: {method}")


def generate_summaries(
    result_version_id: str,
    summarization_config: Dict[str, Any],
    batch_size: int = 50,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate summaries for all articles in the database.

    Args:
        result_version_id: UUID of the result version
        summarization_config: Summarization configuration
        batch_size: Number of articles to process in each batch

    Returns:
        Dictionary with summary statistics
    """
    print(f"Generating summaries using {summarization_config['method']} method...")
    print(f"Target length: {summarization_config['summary_length']}")

    # Initialize summarizer
    summarizer = get_summarizer(summarization_config)
    method = summarization_config["method"]
    summary_length = summarization_config["summary_length"]

    # Statistics
    total_articles = 0
    successful = 0
    failed = 0
    total_time_ms = 0
    total_compression = 0.0

    with Database() as db:
        schema = db.config["schema"]

        # Get articles
        with db.cursor() as cur:
            query = (
                f"SELECT id, title, content "
                f"FROM {schema}.news_articles "
                f"WHERE is_ditwah_cyclone = 1 "
                f"ORDER BY date_posted"
            )
            if limit:
                query += f" LIMIT {int(limit)}"
            cur.execute(query)
            articles = cur.fetchall()
            total_articles = len(articles)

        print(f"Processing {total_articles} articles...")

        # Process articles in batches
        for i in tqdm(range(0, total_articles, batch_size), desc="Summarizing"):
            batch = articles[i:i + batch_size]

            for article in batch:
                article_id = article["id"]
                content = article["content"] or ""
                title = article["title"] or ""

                full_text = content

                if not full_text.strip():
                    failed += 1
                    continue

                start_time = time.time()

                try:
                    summary_text = summarizer.summarize(full_text)

                    if not summary_text:
                        failed += 1
                        continue

                    # Calculate metrics
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    sentence_count = summarizer.count_sentences(summary_text)
                    word_count = summarizer.count_words(summary_text)
                    original_word_count = summarizer.count_words(full_text)

                    # Compression ratio (percentage of original length)
                    compression_ratio = (word_count / original_word_count) if original_word_count > 0 else 0.0

                    # Store summary
                    with db.cursor() as cur:
                        cur.execute(
                            f"""
                            INSERT INTO {schema}.article_summaries
                            (article_id, result_version_id, summary_text, method, summary_length,
                             sentence_count, word_count, compression_ratio, processing_time_ms)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (article_id, result_version_id)
                            DO UPDATE SET
                                summary_text = EXCLUDED.summary_text,
                                method = EXCLUDED.method,
                                summary_length = EXCLUDED.summary_length,
                                sentence_count = EXCLUDED.sentence_count,
                                word_count = EXCLUDED.word_count,
                                compression_ratio = EXCLUDED.compression_ratio,
                                processing_time_ms = EXCLUDED.processing_time_ms,
                                created_at = NOW()
                            """,
                            (
                                article_id,
                                result_version_id,
                                summary_text,
                                method,
                                summary_length,
                                sentence_count,
                                word_count,
                                compression_ratio,
                                processing_time_ms
                            )
                        )

                    successful += 1
                    total_time_ms += processing_time_ms
                    total_compression += compression_ratio

                except Exception as e:
                    print(f"\nError summarizing article {article_id} (title: '{title[:50]}...'): {e}")
                    failed += 1
                    continue

    # Calculate statistics
    avg_time_ms = total_time_ms / successful if successful > 0 else 0
    avg_compression = (total_compression / successful * 100) if successful > 0 else 0

    print(f"\n✓ Summarization complete!")
    print(f"  Total articles: {total_articles}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Avg processing time: {avg_time_ms:.0f}ms")
    print(f"  Avg compression: {avg_compression:.1f}%")

    return {
        "total_articles": total_articles,
        "successful": successful,
        "failed": failed,
        "avg_time_ms": avg_time_ms,
        "avg_compression": avg_compression
    }
