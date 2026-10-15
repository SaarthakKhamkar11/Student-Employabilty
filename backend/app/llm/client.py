"""LLM client boundary. Runtime provider: Gemini (project decision 2026-10-08).

The LLM is a reasoning component only (spec §4, rule 5). It returns small,
schema-constrained decisions, such as a routing plan, and is never asked to
compute a number: numeric work happens in deterministic tools. Every response
is validated locally with Pydantic, whatever the provider claims to enforce.
"""

import asyncio
import logging
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Protocol, TypeVar

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


class LLMQuotaError(LLMError):
    """HTTP 429: the model's quota is used up. retry_after_s is the provider's own reset delay, when it gives one."""

    def __init__(self, message: str, retry_after_s: float | None = None, model: str | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.model = model


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
            if code == 429:  # read the quota details before the long message is redacted
                delay = _RETRY_DELAY.search(message)
                quota = _QUOTA_ID.search(message)
                raise LLMQuotaError(
                    f"{type(exc).__name__}{status}: {self.model_name} quota exhausted"
                    + (f" ({quota.group(1)})" if quota else "") + (f", resets in {delay.group(1)}s" if delay else ""),
                    retry_after_s=float(delay.group(1)) if delay else None, model=self.model_name) from exc
            raise LLMError(f"{type(exc).__name__}{status}: {redact_text(message, 200)}") from exc
        return parse_structured(response.text, schema)


_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")
_QUOTA_ID = re.compile(r"quotaId['\"]?\s*:\s*['\"]?([A-Za-z-]+)")
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
    if llm is None or isinstance(llm, TracedLLMClient) or getattr(llm, "traces_itself", False):
        return llm
    return TracedLLMClient(llm)


_served: ContextVar[dict | None] = ContextVar("llm_served", default=None)


@contextmanager
def capture_served_model() -> Iterator[dict]:
    """Collects which model actually answered the calls made inside the block ({"model": name}). A dict, not a
    value, so a call run as its own task (asyncio.wait_for) still reports back."""
    info: dict = {}
    token = _served.set(info)
    try:
        yield info
    finally:
        _served.reset(token)


class FallbackLLMClient:
    """Quota-aware model chain (free-tier limits are per model, e.g. 20 requests/day for gemini-3.8-flash).

    Each call goes to the first model whose quota isn't known to be exhausted. On HTTP 429 that model is put in
    cooldown until the provider's own reset delay (else cooldown_s) and the next model is tried at once. When every
    model is in cooldown the call fails immediately with LLMQuotaError, without a network call, so callers fall back
    to their deterministic paths without waiting or wasting retries. Every real provider call is still its own trace
    span, under the model that served (or refused) it."""

    traces_itself = True

    def __init__(self, clients: list[LLMClient], cooldown_s: float = 3600.0, clock=time.monotonic,
                 quota_attempts: int = 1, backoff_s: float = 1.0, sleep=asyncio.sleep):
        if not clients:
            raise ValueError("FallbackLLMClient needs at least one client")
        self.clients = [traced(c) for c in clients]
        self.model_name = clients[0].model_name  # the primary; capture_served_model() says who answered
        self.cooldown_s = cooldown_s
        # Phase 13: a 429 is retried on the same model with exponential backoff (backoff_s, 2x, 4x ...) up to
        # quota_attempts calls in total; only then does the model go into cooldown and the chain move on.
        self.quota_attempts = max(1, quota_attempts)
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._clock = clock
        self._until: dict[str, float] = {}

    def status(self) -> list[dict]:
        now = self._clock()
        return [{"model": c.model_name, "available": self._until.get(c.model_name, 0.0) <= now,
                 "cooldown_remaining_s": max(0, round(self._until.get(c.model_name, 0.0) - now))}
                for c in self.clients]

    async def generate_structured(self, *, system: str, prompt: str, schema: type[T], timeout_s: float,
                                  max_output_tokens: int = 1024) -> T:
        now = self._clock()
        ready = [c for c in self.clients if self._until.get(c.model_name, 0.0) <= now]
        if not ready:
            wait = min(self._until.values()) - now
            raise LLMQuotaError(f"every configured model's quota is exhausted; the first resets in {wait:.0f}s",
                                retry_after_s=wait)
        last: LLMQuotaError | None = None
        for client in ready:
            try:
                result = await self._with_backoff(client, system=system, prompt=prompt, schema=schema,
                                                  timeout_s=timeout_s, max_output_tokens=max_output_tokens)
            except LLMQuotaError as exc:
                self._until[client.model_name] = self._clock() + (exc.retry_after_s or self.cooldown_s)
                log_event(_logger, "llm_quota_cooldown", logging.WARNING, model=client.model_name,
                          cooldown_s=round(exc.retry_after_s or self.cooldown_s))
                last = exc
                continue
            if (info := _served.get()) is not None:
                info["model"] = client.model_name
            return result
        raise last


    async def _with_backoff(self, client: LLMClient, **kwargs):
        for attempt in range(1, self.quota_attempts + 1):
            try:
                return await client.generate_structured(**kwargs)
            except LLMQuotaError:
                if attempt == self.quota_attempts:
                    raise
                delay = self.backoff_s * 2 ** (attempt - 1)
                log_event(_logger, "llm_quota_backoff", logging.WARNING, model=client.model_name, attempt=attempt,
                          sleep_s=delay)
                await self._sleep(delay)


def served_model(llm: LLMClient, info: dict) -> str:
    return info.get("model") or llm.model_name


def build_llm_client(settings: Settings) -> LLMClient | None:
    """A configured client, or None, in which case callers use their deterministic path."""
    if not settings.llm_configured:
        return None
    key = settings.gemini_api_key.get_secret_value()
    models = [settings.gemini_model] + [m.strip() for m in (settings.gemini_fallback_models or "").split(",")
                                        if m.strip() and m.strip() != settings.gemini_model]
    return FallbackLLMClient([GeminiClient(key, m) for m in models], cooldown_s=settings.llm_quota_cooldown_seconds,
                             quota_attempts=settings.llm_quota_attempts, backoff_s=settings.llm_quota_backoff_seconds)
