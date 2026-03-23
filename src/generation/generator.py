from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Optional


class BaseGenerator:
    """Abstract text generator interface."""

    def generate(self, *, prompt: str) -> str:
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """Simple generator that extracts `Answer:` from the prompt context if present."""

    _answer_re = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE)

    def generate(self, *, prompt: str) -> str:
        # The baseline dataset includes an explicit `Answer:` token in the document text.
        m = self._answer_re.search(prompt)
        if not m:
            return "I don't know"
        ans = m.group(1).strip()
        # Remove trailing punctuation for cleaner EM/F1.
        ans = ans.rstrip(".!?")
        return ans


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    api_base: str
    api_key: str
    max_tokens: int = 128
    temperature: float = 0.0
    timeout_s: int = 60


class OpenAICompatibleGenerator(BaseGenerator):
    """Minimal OpenAI-compatible chat completions client using only stdlib."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    def generate(self, *, prompt: str) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except Exception as e:  # pragma: no cover (network failures)
            raise RuntimeError(f"OpenAI-compatible request failed: {e}") from e

        data = json.loads(body)
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Unexpected API response shape: {data}") from e


def generator_from_config(*, generator_type: str, generation_config: dict) -> BaseGenerator:
    """Create a generator instance from config.

    - generator_type='mock' always works.
    - generator_type='openai_compatible' requires OPENAI-compatible env vars.
    """
    if generator_type == "mock":
        return MockGenerator()

    if generator_type == "openai_compatible":
        model_env = generation_config.get("openai_compatible", {}).get("model_env", "OPENAI_MODEL")
        api_base_env = generation_config.get("openai_compatible", {}).get("api_base_env", "OPENAI_API_BASE")
        api_key_env = generation_config.get("openai_compatible", {}).get("api_key_env", "OPENAI_API_KEY")
        default_model = generation_config.get("openai_compatible", {}).get("default_model", "gpt-4o-mini")

        model = os.getenv(model_env, default_model)
        api_base = os.getenv(api_base_env, "https://api.openai.com/v1")
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Missing API key for openai_compatible generator. Set env var {api_key_env}."
            )

        return OpenAICompatibleGenerator(
            OpenAICompatibleConfig(
                model=model,
                api_base=api_base,
                api_key=api_key,
                max_tokens=int(generation_config.get("max_tokens", 128)),
                temperature=float(generation_config.get("temperature", 0.0)),
            )
        )

    raise ValueError(f"Unsupported generator_type: {generator_type}")

