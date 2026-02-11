"""LLM client abstraction supporting multiple providers."""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.config import load_config


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    usage: Dict[str, int]
    model: str
    provider: str


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        response = self.generate(prompt, system_prompt, json_mode=True)
        return json.loads(response.content)


class ClaudeLLM(BaseLLM):
    """Anthropic Claude client."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(model, **kwargs)
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]

        if json_mode and system_prompt:
            system_prompt += "\n\nRespond only with valid JSON, no other text."
        elif json_mode:
            system_prompt = "Respond only with valid JSON, no other text."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "",
            messages=messages
        )

        return LLMResponse(
            content=response.content[0].text,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            model=self.model,
            provider="claude"
        )


class OpenAILLM(BaseLLM):
    """OpenAI GPT client."""

    def __init__(self, model: str = "gpt-4o", **kwargs):
        super().__init__(model, **kwargs)
        from openai import OpenAI

        # Try to get API key from config first, then environment variable
        api_key = None
        try:
            config = load_config()
            api_key = config.get("openai", {}).get("api_key") or config.get("llm", {}).get("api_key")
        except:
            pass

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in config.yaml under 'openai.api_key' "
                "or as environment variable OPENAI_API_KEY"
            )
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            },
            model=self.model,
            provider="openai"
        )


class MistralLLM(BaseLLM):
    """Mistral AI client with rate limit retry logic."""

    def __init__(self, model: str = "mistral-large-latest", **kwargs):
        super().__init__(model, **kwargs)
        from mistralai import Mistral
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        self.client = Mistral(api_key=api_key)

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_retries: int = 5
    ) -> Any:
        """
        Call Mistral API with exponential backoff retry logic for rate limiting.

        Args:
            messages: List of message dicts with role and content
            json_mode: Whether to use JSON response format
            max_retries: Maximum number of retry attempts

        Returns:
            API response object

        Raises:
            Exception: After max retries exceeded
        """
        from mistralai.models import SDKError

        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }

                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.complete(**kwargs)
                return response

            except SDKError as e:
                # Check for rate limit error (429)
                if hasattr(e, 'status_code') and e.status_code == 429:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        wait_time = 2 ** attempt
                        print(f"  Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"Max retries ({max_retries}) exceeded for rate limiting")
                else:
                    # Non-rate-limit error, raise immediately
                    raise
            except Exception as e:
                # Unknown error, raise immediately
                raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if json_mode and system_prompt is None:
            # Add JSON instruction if not already in system prompt
            messages.insert(0, {"role": "system", "content": "Respond only with valid JSON, no other text."})

        response = self._call_with_retry(messages, json_mode=json_mode)

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            },
            model=self.model,
            provider="mistral"
        )


class LocalLLM(BaseLLM):
    """Local LLM client (Ollama-compatible API)."""

    def __init__(
        self,
        model: str = "llama3.1:70b",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        import requests

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        if json_mode:
            full_prompt += "\n\nRespond only with valid JSON."

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["response"],
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0)
            },
            model=self.model,
            provider="local"
        )


class GeminiLLM(BaseLLM):
    """Google Gemini client."""

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model, **kwargs)
        from google import genai
        from google.genai import types

        # Try to get API key from config first, then environment variable
        api_key = None
        try:
            config = load_config()
            api_key = config.get("gemini", {}).get("api_key") or config.get("google", {}).get("api_key")
        except:
            pass

        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in config.yaml under 'gemini.api_key' "
                "or as environment variable GOOGLE_API_KEY"
            )

        self.client = genai.Client(api_key=api_key)
        self.types = types

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        if json_mode:
            full_prompt += "\n\nRespond only with valid JSON, no other text."

        # Configure generation parameters
        generation_config = self.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=generation_config
        )

        # Extract usage information
        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
            "output_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
        }

        return LLMResponse(
            content=response.text,
            usage=usage,
            model=self.model,
            provider="gemini"
        )


class EmbeddingClient:
    """Embedding client supporting OpenAI and local models."""

    def __init__(
        self,
        provider: str = "local",
        model: str = "all-mpnet-base-v2",
        dimensions: Optional[int] = None,
        task: Optional[str] = None,
        matryoshka_dim: Optional[int] = None
    ):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.task = task  # For EmbeddingGemma: "clustering", "classification", "retrieval"
        self.matryoshka_dim = matryoshka_dim  # 768, 512, 256, or 128
        self.is_embeddinggemma = "embeddinggemma" in model.lower()

        if provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
        else:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {model}...")
            self.client = SentenceTransformer(model)
            self.dimensions = self.client.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimensions: {self.dimensions}")

            if self.is_embeddinggemma and self.task:
                print(f"Using EmbeddingGemma with task: {self.task}")
            if self.matryoshka_dim:
                print(f"Matryoshka truncation enabled: {self.matryoshka_dim} dimensions")

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.provider == "openai":
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                kwargs = {
                    "model": self.model,
                    "input": batch
                }
                # Only pass dimensions if explicitly specified
                if self.dimensions is not None:
                    kwargs["dimensions"] = self.dimensions
                response = self.client.embeddings.create(**kwargs)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        else:
            # Local model
            if self.is_embeddinggemma and self.task:
                # Use task-specific prompting for EmbeddingGemma
                task_prompt = self._get_task_prompt()
                embeddings = self.client.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    prompt=task_prompt
                )
            else:
                # Standard encoding for other models
                embeddings = self.client.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )

            # Apply Matryoshka truncation if specified
            if self.matryoshka_dim and self.matryoshka_dim < embeddings.shape[1]:
                embeddings = self._truncate_and_normalize(embeddings, self.matryoshka_dim)

            return embeddings.tolist()

    def _get_task_prompt(self) -> str:
        """Get EmbeddingGemma task prompt based on analysis type."""
        task_prompts = {
            "clustering": "task: clustering | query: ",
            "classification": "task: classification | query: ",
            "retrieval": "task: search result | query: ",
            "similarity": "task: sentence similarity | query: "
        }
        return task_prompts.get(self.task, "task: classification | query: ")

    def _truncate_and_normalize(self, embeddings, target_dim):
        """Truncate to target dimension and re-normalize (Matryoshka)."""
        import numpy as np
        truncated = embeddings[:, :target_dim]
        # Re-normalize to unit length
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        normalized = truncated / (norms + 1e-8)
        return normalized

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]


def get_llm(config: dict = None) -> BaseLLM:
    """Factory function to get LLM based on config."""
    if config is None:
        config = load_config()["llm"]

    provider = config.get("provider", "claude")
    model = config.get("model")
    temperature = config.get("temperature", 0.0)
    max_tokens = config.get("max_tokens", 4096)

    if provider == "claude":
        return ClaudeLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "openai":
        return OpenAILLM(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "gemini":
        return GeminiLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "mistral":
        return MistralLLM(model=model, temperature=temperature, max_tokens=max_tokens)
    elif provider == "local":
        base_url = config.get("base_url", "http://localhost:11434")
        return LocalLLM(model=model, base_url=base_url, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_embeddings_client(config: dict = None, analysis_type: str = None) -> EmbeddingClient:
    """Factory function to get embedding client."""
    if config is None:
        config = load_config()["embeddings"]

    # Auto-detect task from analysis type if not specified
    task = config.get("task")
    if task is None and analysis_type:
        task_mapping = {
            "topics": "classification",
            "clustering": "clustering",
            "summarization": "retrieval"
        }
        task = task_mapping.get(analysis_type, "classification")

    print(f"Creating EmbeddingClient with provider: {config.get('provider', 'local')}, model: {config.get('model', 'all-mpnet-base-v2')}, task: {task}")
    return EmbeddingClient(
        provider=config.get("provider", "local"),
        model=config.get("model", "all-mpnet-base-v2"),
        dimensions=config.get("dimensions"),  # None if not specified (auto-detect for local models)
        task=task,
        matryoshka_dim=config.get("matryoshka_dim")
    )
