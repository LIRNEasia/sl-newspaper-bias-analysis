"""Multi-document summarization for topic groups and event clusters."""

import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple

# Resolve the project root (parent of src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.llm import get_llm


class MultiDocSummarizer(ABC):
    """Abstract base class for multi-document summarizers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-document summarizer with configuration.

        Args:
            config: Summarization configuration dictionary
        """
        self.config = config
        self.method = config.get("method", "primera")
        self.summary_length = config.get("summary_length", "medium")

    @abstractmethod
    def summarize_multiple(self, documents: List[str], sources: List[str] = None) -> str:
        """
        Generate summary from multiple documents.

        Args:
            documents: List of article texts
            sources: Optional list of source names corresponding to documents

        Returns:
            Consolidated summary
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
                self.config.get("short_sentences", 5),
                self.config.get("short_words", 80)
            )
        elif length == "long":
            return (
                self.config.get("long_sentences", 12),
                self.config.get("long_words", 200)
            )
        else:  # medium
            return (
                self.config.get("medium_sentences", 8),
                self.config.get("medium_words", 150)
            )

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())


class PRIMERASummarizer(MultiDocSummarizer):
    """Multi-document summarization using PRIMERA model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "allenai/primera"
        try:
            # Force CPU processing to avoid CUDA issues
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
            self.max_tokens = 4096  # PRIMERA's max input length
            print(f"Loaded {model_name} on CPU (max {self.max_tokens} tokens)")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def summarize_multiple(self, documents: List[str], sources: List[str] = None) -> str:
        """Generate PRIMERA multi-document summary."""
        if not documents or not any(doc.strip() for doc in documents):
            return ""

        try:
            # Join documents with PRIMERA's document separator token
            combined_text = " <doc-sep> ".join(doc.strip() for doc in documents if doc.strip())

            # Check for truncation before tokenizing
            token_ids = self.tokenizer.encode(combined_text, add_special_tokens=False)
            token_count = len(token_ids)

            if token_count > self.max_tokens:
                tokens_lost = token_count - self.max_tokens
                percent_lost = (tokens_lost / token_count) * 100
                print(f"⚠️  WARNING: PRIMERA truncation detected!")
                print(f"   Input: {token_count:,} tokens ({self.count_words(combined_text):,} words)")
                print(f"   Limit: {self.max_tokens:,} tokens")
                print(f"   Lost: {tokens_lost:,} tokens (~{percent_lost:.1f}% of content)")
                print(f"   Suggestion: Use LED (16K tokens) or Claude/Gemini for longer inputs")

            # Tokenize and truncate to max length
            inputs = self.tokenizer(
                combined_text,
                max_length=self.max_tokens,
                truncation=True,
                return_tensors="pt"
            )

            # Generate summary - allow model to generate freely within reasonable bounds
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=512,      # allow up to ~400 words output
                min_length=100,      # force at least ~75 words
                num_beams=4,
                length_penalty=1.0,  # neutral - don't penalize length
                early_stopping=True,
                no_repeat_ngram_size=3
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            print(f"PRIMERA error: {e}")
            return ""


class LEDMultiDocSummarizer(MultiDocSummarizer):
    """Multi-document summarization using LED (Longformer Encoder-Decoder)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from transformers import LEDTokenizer, LEDForConditionalGeneration

        model_name = "allenai/led-base-16384"
        try:
            self.tokenizer = LEDTokenizer.from_pretrained(model_name)
            self.model = LEDForConditionalGeneration.from_pretrained(model_name)
            self.model.eval()
            self.max_tokens = 16384  # LED's long context
            print(f"Loaded {model_name} on CPU (max {self.max_tokens} tokens)")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def summarize_multiple(self, documents: List[str], sources: List[str] = None) -> str:
        """Generate LED multi-document summary."""
        if not documents or not any(doc.strip() for doc in documents):
            return ""

        try:
            # Add source labels if provided
            if sources and len(sources) == len(documents):
                labeled_docs = [
                    f"[{source}] {doc.strip()}"
                    for source, doc in zip(sources, documents)
                    if doc.strip()
                ]
                combined_text = "\n\n".join(labeled_docs)
            else:
                combined_text = "\n\n".join(doc.strip() for doc in documents if doc.strip())

            # Check for truncation before tokenizing
            token_ids = self.tokenizer.encode(combined_text, add_special_tokens=False)
            token_count = len(token_ids)

            if token_count > self.max_tokens:
                tokens_lost = token_count - self.max_tokens
                percent_lost = (tokens_lost / token_count) * 100
                print(f"⚠️  WARNING: LED truncation detected!")
                print(f"   Input: {token_count:,} tokens ({self.count_words(combined_text):,} words)")
                print(f"   Limit: {self.max_tokens:,} tokens")
                print(f"   Lost: {tokens_lost:,} tokens (~{percent_lost:.1f}% of content)")
                print(f"   Suggestion: Use Claude/Gemini for inputs >{self.max_tokens:,} tokens")

            # Tokenize and truncate
            inputs = self.tokenizer(
                combined_text,
                max_length=self.max_tokens,
                truncation=True,
                return_tensors="pt"
            )

            # Set global attention on first token (LED requirement)
            global_attention_mask = inputs["attention_mask"].clone()
            global_attention_mask[:, 0] = 1

            # Generate summary - allow model to generate freely within reasonable bounds
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                global_attention_mask=global_attention_mask,
                max_length=512,      # allow up to ~400 words output
                min_length=100,      # force at least ~75 words
                num_beams=4,
                length_penalty=1.0,  # neutral - don't penalize length
                early_stopping=True,
                no_repeat_ngram_size=3
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            print(f"LED error: {e}")
            return ""


class LongT5MultiDocSummarizer(MultiDocSummarizer):
    """Multi-document summarization using LongT5."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "google/long-t5-tglobal-base"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.eval()
            self.max_tokens = 4096  # LongT5 max length
            print(f"Loaded {model_name} on CPU (max {self.max_tokens} tokens)")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def summarize_multiple(self, documents: List[str], sources: List[str] = None) -> str:
        """Generate LongT5 multi-document summary."""
        if not documents or not any(doc.strip() for doc in documents):
            return ""

        try:
            # Add source labels if provided
            if sources and len(sources) == len(documents):
                labeled_docs = [
                    f"[{source}] {doc.strip()}"
                    for source, doc in zip(sources, documents)
                    if doc.strip()
                ]
                combined_text = "\n\n".join(labeled_docs)
            else:
                combined_text = "\n\n".join(doc.strip() for doc in documents if doc.strip())

            # Check for truncation before tokenizing
            token_ids = self.tokenizer.encode(combined_text, add_special_tokens=False)
            token_count = len(token_ids)

            if token_count > self.max_tokens:
                tokens_lost = token_count - self.max_tokens
                percent_lost = (tokens_lost / token_count) * 100
                print(f"⚠️  WARNING: LongT5 truncation detected!")
                print(f"   Input: {token_count:,} tokens ({self.count_words(combined_text):,} words)")
                print(f"   Limit: {self.max_tokens:,} tokens")
                print(f"   Lost: {tokens_lost:,} tokens (~{percent_lost:.1f}% of content)")
                print(f"   Suggestion: Use LED (16K tokens) or Claude/Gemini for longer inputs")

            # Tokenize and truncate
            inputs = self.tokenizer(
                combined_text,
                max_length=self.max_tokens,
                truncation=True,
                return_tensors="pt"
            )

            # Generate summary - allow model to generate freely within reasonable bounds
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=512,      # allow up to ~400 words output
                min_length=100,      # force at least ~75 words
                num_beams=4,
                length_penalty=1.0,  # neutral - don't penalize length
                early_stopping=True,
                no_repeat_ngram_size=3
            )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()

        except Exception as e:
            print(f"LongT5 error: {e}")
            return ""


class OpenAIMultiDocSummarizer(MultiDocSummarizer):
    """Multi-document summarization using OpenAI GPT models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        llm_model = config.get("llm_model", "gpt-4o")
        llm_temperature = config.get("llm_temperature", 0.0)

        llm_config = {
            "provider": "openai",
            "model": llm_model,
            "temperature": llm_temperature,
            "max_tokens": 4096
        }

        self.llm = get_llm(llm_config)

    def summarize_multiple(self, documents: List[str], sources: List[str] = None) -> str:
        """Generate OpenAI multi-document summary."""
        if not documents or not any(doc.strip() for doc in documents):
            return ""

        target_sentences, target_words = self.get_target_length()

        # Build prompt with source attribution
        articles_section = ""
        for i, doc in enumerate(documents):
            if not doc.strip():
                continue
            source_label = sources[i] if sources and i < len(sources) else f"Article {i+1}"
            articles_section += f"\n[{source_label}]\n{doc.strip()}\n"

        prompt = f"""You are summarizing multiple news articles that cover the same topic or event.

Generate a consolidated summary ({target_sentences}-{target_sentences+2} sentences, approximately {target_words} words) that:
- Captures the main points across all articles
- Highlights areas of agreement and disagreement between sources
- Notes which sources emphasized which aspects (when relevant)
- Synthesizes information rather than simply concatenating

Articles:
{articles_section}

Consolidated Summary:"""

        try:
            response = self.llm.generate(prompt)
            return response.content.strip()

        except Exception as e:
            print(f"OpenAI multi-doc error: {e}")
            return ""


class GeminiMultiDocSummarizer(MultiDocSummarizer):
    """Multi-document summarization using Gemini."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        llm_model = config.get("llm_model", "gemini-2.0-flash")
        llm_temperature = config.get("llm_temperature", 0.0)

        llm_config = {
            "provider": "gemini",
            "model": llm_model,
            "temperature": llm_temperature,
            "max_tokens": 4096
        }

        self.llm = get_llm(llm_config)

    def summarize_multiple(self, documents: List[str], sources: List[str] = None) -> str:
        """Generate Gemini multi-document summary."""
        if not documents or not any(doc.strip() for doc in documents):
            return ""

        target_sentences, target_words = self.get_target_length()

        # Build prompt with source attribution
        articles_section = ""
        for i, doc in enumerate(documents):
            if not doc.strip():
                continue
            source_label = sources[i] if sources and i < len(sources) else f"Article {i+1}"
            articles_section += f"\n[{source_label}]\n{doc.strip()}\n"

        prompt = f"""You are summarizing multiple news articles that cover the same topic or event.

Generate a consolidated summary ({target_sentences}-{target_sentences+2} sentences, approximately {target_words} words) that:
- Captures the main points across all articles
- Highlights areas of agreement and disagreement between sources
- Notes which sources emphasized which aspects (when relevant)
- Synthesizes information rather than simply concatenating

Articles:
{articles_section}

Consolidated Summary:"""

        try:
            response = self.llm.generate(prompt)
            return response.content.strip()

        except Exception as e:
            print(f"Gemini multi-doc error: {e}")
            return ""


def get_multi_doc_summarizer(config: Dict[str, Any]) -> MultiDocSummarizer:
    """
    Factory function to get the appropriate multi-document summarizer.

    Args:
        config: Summarization configuration

    Returns:
        MultiDocSummarizer instance

    Raises:
        ValueError: If method is not supported
    """
    method = config.get("method", "primera").lower()

    if method == "primera":
        return PRIMERASummarizer(config)
    elif method == "led":
        return LEDMultiDocSummarizer(config)
    elif method == "longt5":
        return LongT5MultiDocSummarizer(config)
    elif method in ["openai", "gpt", "gpt-4", "gpt-4o"]:
        return OpenAIMultiDocSummarizer(config)
    elif method == "gemini":
        return GeminiMultiDocSummarizer(config)
    else:
        raise ValueError(f"Unsupported multi-doc summarization method: {method}")
