import json
import logging

from pydantic import SecretStr

from app.core.logging import REDACTED, JsonFormatter, get_logger, log_event, redact, request_id_var


def test_sensitive_keys_are_masked_at_any_depth():
    out = redact({"student": {"name": "Asha", "email": "a@b.co", "skills": ["python"]},
                  "gemini_api_key": "fake-gemini-key", "db_password": "pw", "tool_name": "parse_resume"})
    assert out == {"student": {"name": REDACTED, "email": REDACTED, "skills": ["python"]},
                   "gemini_api_key": REDACTED, "db_password": REDACTED, "tool_name": "parse_resume"}


def test_emails_and_phone_numbers_inside_text_are_replaced():
    assert redact("contact asha.k@college.edu or +91 98765 43210") == "contact [EMAIL] or [PHONE]"


def test_ids_and_short_numbers_survive_redaction():
    assert redact("S00001 J0000001 8.75 2027") == "S00001 J0000001 8.75 2027"


def test_long_free_text_is_never_logged_verbatim():
    assert redact("Experienced developer " * 50).startswith("[TEXT len=")


def test_secret_values_are_masked():
    assert redact(SecretStr("x")) == REDACTED


def test_json_formatter_emits_request_id_and_redacted_fields(caplog):
    caplog.set_level(logging.INFO, logger="employability")
    token = request_id_var.set("req-123")
    try:
        log_event(get_logger("test"), "evt", email="x@y.com", count=2)
    finally:
        request_id_var.reset(token)
    entry = json.loads(JsonFormatter().format(caplog.records[-1]))
    assert entry["request_id"] == "req-123" and entry["event"] == "evt"
    assert entry["email"] == REDACTED and entry["count"] == 2
