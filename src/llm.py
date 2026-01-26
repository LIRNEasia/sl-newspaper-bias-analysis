"""LLM client abstraction supporting multiple providers."""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import yaml
from pathlib import Path


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


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
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
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


class EmbeddingClient:
    """Embedding client supporting OpenAI and local models."""

    def __init__(
        self,
        provider: str = "local",
        model: str = "all-mpnet-base-v2",
        dimensions: int = 768
    ):
        self.provider = provider
        self.model = model
        self.dimensions = dimensions

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

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.provider == "openai":
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            return all_embeddings
        else:
            # Local model
            embeddings = self.client.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings.tolist()

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
    elif provider == "local":
        base_url = config.get("base_url", "http://localhost:11434")
        return LocalLLM(model=model, base_url=base_url, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_embeddings_client(config: dict = None) -> EmbeddingClient:
    """Factory function to get embedding client."""
    if config is None:
        config = load_config()["embeddings"]

    return EmbeddingClient(
        provider=config.get("provider", "local"),
        model=config.get("model", "all-mpnet-base-v2"),
        dimensions=config.get("dimensions", 768)
    )
