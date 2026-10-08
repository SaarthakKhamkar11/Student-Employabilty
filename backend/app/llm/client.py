"""LLM client boundary. Runtime provider: Gemini (project decision 2026-10-08).

The LLM is a reasoning component only (spec §4, rule 5). It returns small,
schema-constrained decisions, such as a routing plan, and is never asked to
compute a number: numeric work happens in deterministic tools. Every response
is validated locally with Pydantic, whatever the provider claims to enforce.
"""

import asyncio
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import redact_text

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM call failed: network, auth, quota or provider error."""


class LLMTimeoutError(LLMError):
    """The LLM did not answer within the configured timeout."""


class LLMInvalidOutputError(LLMError):
    """The LLM answered, but not with JSON matching the requested schema."""


class LLMClient(Protocol):
    model_name: str

    async def generate_structured(self, *, system: str, prompt: str, schema: type[T], timeout_s: float) -> T: ...


def parse_structured(text: str | None, schema: type[T]) -> T:
    """Validate raw LLM text against schema. Shared by real and test clients."""
    if not text:
        raise LLMInvalidOutputError("empty response")
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMInvalidOutputError(f"response failed {schema.__name__} validation ({exc.error_count()} error(s))") from exc


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.model_name = model
        self._api_key = api_key  # kept only to scrub it from error messages
        self._client = genai.Client(api_key=api_key)

    async def generate_structured(self, *, system: str, prompt: str, schema: type[T], timeout_s: float) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            # The SDK must not retry on its own: the caller owns the retry budget.
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000),
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )
        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(model=self.model_name, contents=prompt, config=config),
                timeout_s,
            )
        except TimeoutError as exc:
            raise LLMTimeoutError(f"no response within {timeout_s}s") from exc
        except Exception as exc:
            message = str(exc).replace(self._api_key, "[REDACTED]")
            raise LLMError(f"{type(exc).__name__}: {redact_text(message, 200)}") from exc
        return parse_structured(response.text, schema)


def build_llm_client(settings: Settings) -> LLMClient | None:
    """A configured client, or None, in which case callers use their deterministic path."""
    if not settings.llm_configured:
        return None
    return GeminiClient(settings.gemini_api_key.get_secret_value(), settings.gemini_model)
