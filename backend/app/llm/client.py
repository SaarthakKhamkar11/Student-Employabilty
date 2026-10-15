"""LLM client boundary. Runtime provider: Gemini (project decision 2026-10-08).

The LLM is a reasoning component only (spec §4, rule 5). It returns small,
schema-constrained decisions, such as a routing plan, and is never asked to
compute a number: numeric work happens in deterministic tools. Every response
is validated locally with Pydantic, whatever the provider claims to enforce.
"""

import asyncio
import logging
import re
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import get_logger, log_event, redact_text
from app.core.tracing import current_trace, open_span

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """The LLM call failed: network, auth, quota or provider error."""


class LLMTimeoutError(LLMError):
    """The LLM did not answer within the configured timeout."""


class LLMInvalidOutputError(LLMError):
    """The LLM answered, but not with JSON matching the requested schema."""


class LLMClient(Protocol):
    model_name: str

    async def generate_structured(self, *, system: str, prompt: str, schema: type[T], timeout_s: float,
                                  max_output_tokens: int = 1024) -> T: ...


def parse_structured(text: str | None, schema: type[T]) -> T:
    """Validate raw LLM text against schema. Shared by real and test clients."""
    if not text:
        raise LLMInvalidOutputError("empty response")
    try:
        return schema.model_validate_json(text)
    except ValidationError as exc:
        raise LLMInvalidOutputError(f"response failed {schema.__name__} validation ({exc.error_count()} error(s))") from exc


# Constraints Gemini's response schema may reject; Pydantic still enforces every one of them on the reply.
_LOCAL_ONLY = {"$defs", "title", "minLength", "maxLength", "minItems", "maxItems", "pattern", "default"}


def response_schema(schema: type[BaseModel]) -> dict:
    """The schema sent to the provider: nested models inlined (no $defs/$ref) and length/count limits left to the
    local validation. A nested schema with both was refused with 400 INVALID_ARGUMENT (Phase 11, ADR 0019)."""
    full = schema.model_json_schema()
    defs = full.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
            out = {k: walk(v) for k, v in node.items() if k not in _LOCAL_ONLY and k != "properties"}
            if "properties" in node:  # property names are data, never keywords to drop
                out["properties"] = {name: walk(sub) for name, sub in node["properties"].items()}
            return out
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(full)


class GeminiClient:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.model_name = model
        self._api_key = api_key  # kept only to scrub it from error messages
        self._client = genai.Client(api_key=api_key)

    async def generate_structured(self, *, system: str, prompt: str, schema: type[T], timeout_s: float,
                                  max_output_tokens: int = 1024) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=response_schema(schema),
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
            code = getattr(exc, "code", None)  # the HTTP status survives the redaction of a long message
            status = f" {code} {getattr(exc, 'status', '') or ''}".rstrip() if isinstance(code, int) else ""
            raise LLMError(f"{type(exc).__name__}{status}: {redact_text(message, 200)}") from exc
        return parse_structured(response.text, schema)


_HTTP_STATUS = re.compile(r"(?<![\d.])([45]\d\d)(?![\d.])")  # "ClientError 429 RESOURCE_EXHAUSTED: ..."
_logger = get_logger("llm")


class TracedLLMClient:
    """Wraps any LLMClient so every provider call is traced (Phase 12, spec §39): which agent spent it, the model,
    the latency, and how it ended (ok, timeout, invalid output, or an HTTP status such as 429 for the quota).
    The prompt and the reply are never recorded."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.model_name = inner.model_name

    async def generate_structured(self, *, system: str, prompt: str, schema: type[T], timeout_s: float,
                                  max_output_tokens: int = 1024) -> T:
        with open_span("llm", self.model_name, purpose=schema.__name__) as span:
            try:
                result = await self.inner.generate_structured(system=system, prompt=prompt, schema=schema,
                                                              timeout_s=timeout_s, max_output_tokens=max_output_tokens)
            except Exception as exc:
                status = _HTTP_STATUS.search(str(exc)) if isinstance(exc, LLMError) else None
                self._record(span, "failed", error_type=type(exc).__name__,
                             http_status=int(status.group(1)) if status else None)
                raise
            self._record(span, "success", error_type=None, http_status=None)
            return result

    def _record(self, span, status: str, **attrs) -> None:
        trace = current_trace()
        if span is not None and trace is not None:
            trace.finish(span, status, **attrs)
            owner = trace.owning_agent(span)
            span.attrs["agent_name"] = owner.name if owner else None
        log_event(_logger, "llm_call", logging.INFO if status == "success" else logging.WARNING,
                  model=self.model_name, purpose=span.attrs.get("purpose") if span else None, status=status,
                  agent_name=span.attrs.get("agent_name") if span else None,
                  duration_ms=span.duration_ms if span else None, **attrs)


def traced(llm: LLMClient | None) -> LLMClient | None:
    if llm is None or isinstance(llm, TracedLLMClient):
        return llm
    return TracedLLMClient(llm)


def build_llm_client(settings: Settings) -> LLMClient | None:
    """A configured client, or None, in which case callers use their deterministic path."""
    if not settings.llm_configured:
        return None
    return GeminiClient(settings.gemini_api_key.get_secret_value(), settings.gemini_model)
