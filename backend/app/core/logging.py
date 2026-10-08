"""Structured JSON logging with PII redaction.

Every record is one JSON object tagged with the current request_id, which the
agent runtime sets in a context variable. Fields pass through redact() before
they reach a handler:
- values under sensitive keys (email, phone, password, api_key, resume_text ...)
  are masked;
- email addresses and phone-number-like digit runs inside strings are replaced;
- long free text, such as resume or JD bodies, is never logged verbatim.

This is the safe baseline. Full hardening is Phase 16 (security).
"""

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, SecretStr

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

REDACTED = "[REDACTED]"
MAX_LOGGED_TEXT = 120  # longer strings are summarised, never logged verbatim
MAX_LOGGED_ITEMS = 50  # longer lists (e.g. embedding vectors) are summarised

# Personal data, secrets and raw documents: masked everywhere (logs and persisted records).
PII_KEYS = {
    "name", "full_name", "first_name", "last_name", "student_name",
    "email", "phone", "phone_number", "mobile", "address",
    "password", "otp", "token", "access_token", "refresh_token", "jwt",
    "secret", "jwt_secret", "api_key", "gemini_api_key", "authorization",
    "cookie", "database_url", "resume_text", "jd_text", "raw_text", "content_b64", "resume_file_b64",
}
# Pipeline payloads: kept in persisted results, but never written to logs (the PII detector logs counts).
LOG_ONLY_KEYS = {"text", "fields", "masked_fields", "record", "profile", "sections", "section_texts", "vectors"}
SENSITIVE_KEYS = PII_KEYS | LOG_ONLY_KEYS
SENSITIVE_KEY_PARTS = ("password", "secret", "token", "api_key", "apikey")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?!\w)(?!\.\d)")  # sentence-final "." allowed

_ROOT_LOGGER = "employability"


def _is_sensitive_key(key: str, keys: set[str] = SENSITIVE_KEYS) -> bool:
    k = key.lower()
    return k in keys or any(part in k for part in SENSITIVE_KEY_PARTS)


def redact_text(text: str, max_text: int | None = MAX_LOGGED_TEXT) -> str:
    if max_text is not None and len(text) > max_text:
        return f"[TEXT len={len(text)}]"
    return PHONE_RE.sub("[PHONE]", EMAIL_RE.sub("[EMAIL]", text))


def redact(value: Any, key: str | None = None, *, max_text: int | None = MAX_LOGGED_TEXT,
           keys: set[str] = SENSITIVE_KEYS) -> Any:
    """Return a copy of value that is safe to log.

    Persistence calls this with max_text=None and keys=PII_KEYS: long text and pipeline
    payloads are kept, and only PII and secrets are masked.
    """
    if key is not None and _is_sensitive_key(key, keys):
        return REDACTED
    if isinstance(value, SecretStr):
        return REDACTED
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: redact(v, str(k), max_text=max_text, keys=keys) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        if max_text is not None and len(value) > MAX_LOGGED_ITEMS:
            return f"[LIST len={len(value)}]"
        return [redact(v, max_text=max_text, keys=keys) for v in value]
    if isinstance(value, str):
        return redact_text(value, max_text)
    return value


def summarize_arguments(value: Any, key: str | None = None) -> Any:
    """The SHAPE of tool arguments, never their text (Phase 5).

    Strings become "[str len=N]" and lists "[list len=N]"; numbers, booleans and None are kept; sensitive
    keys are masked. Safe by default: a new tool with unlisted keys such as "posting" or "description"
    cannot leak document text into the logs.
    """
    if key is not None and _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: summarize_arguments(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return f"[list len={len(value)}]"
    if isinstance(value, str):
        return f"[str len={len(value)}]"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", None),
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(entry, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Attach the JSON handler once. Safe to call repeatedly."""
    logger = logging.getLogger(_ROOT_LOGGER)
    logger.setLevel(level.upper())
    if not any(getattr(h, "_employability", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._employability = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_LOGGER}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log a structured event. Fields are redacted here, before any handler sees them."""
    logger.log(level, event, extra={"fields": redact(fields), "request_id": request_id_var.get()})
