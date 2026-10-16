"""Redis-backed OTP state, resend limits, single-use signup tokens and login-failure counters.

Keys (all expire on their own):
  auth:otp:{email}            hash {code_hash, role, attempts}   TTL = OTP_EXPIRY_SECONDS
  auth:otp_cooldown:{email}   "1"                                 TTL = OTP_RESEND_COOLDOWN_SECONDS
  auth:otp_sends:{email}      counter                             TTL = 1 hour
  auth:signup_jti:{jti}       email                               TTL = SIGNUP_TOKEN_EXPIRE_MINUTES
  auth:login_fail:{email}     counter                             TTL = LOGIN_FAILURE_WINDOW_SECONDS

A password-reset code uses the same three OTP keys under the scope "reset_otp" (auth:reset_otp:{email}, ...), so a
reset and a signup never overwrite each other's code or share a resend limit.
"""
from enum import Enum
from functools import lru_cache

import redis

from .config import get_settings
from .security import otp_matches


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _k(kind: str, ident: str) -> str:
    return f"auth:{kind}:{ident}"


def _incr_with_window(key: str, window_seconds: int) -> int:
    """INCR a counter; the expiry is set only when the window starts (works on any Redis version)."""
    r = get_redis()
    value = r.incr(key)
    if value == 1 or r.ttl(key) == -1:
        r.expire(key, window_seconds)
    return value


# ---------------------------------------------------------------- sending limits
SIGNUP = "otp"
RESET = "reset_otp"


def resend_wait_seconds(email: str, scope: str = SIGNUP) -> int:
    """>0 means the user must wait this many seconds before another OTP."""
    ttl = get_redis().ttl(_k(f"{scope}_cooldown", email))
    return max(ttl, 0)


def sends_this_hour(email: str, scope: str = SIGNUP) -> int:
    v = get_redis().get(_k(f"{scope}_sends", email))
    return int(v) if v else 0


def save_otp(email: str, code_hash: str, role: str, scope: str = SIGNUP) -> None:
    s = get_settings()
    r = get_redis()
    pipe = r.pipeline()
    pipe.delete(_k(scope, email))                       # a new OTP replaces any older one
    pipe.hset(_k(scope, email), mapping={"code_hash": code_hash, "role": role, "attempts": 0})
    pipe.expire(_k(scope, email), s.otp_expiry_seconds)
    pipe.set(_k(f"{scope}_cooldown", email), "1", ex=s.otp_resend_cooldown_seconds)
    pipe.execute()
    _incr_with_window(_k(f"{scope}_sends", email), 3600)  # hourly window starts at first send


def discard_otp(email: str, scope: str = SIGNUP) -> None:
    """Used when the email failed to send, so the user can retry immediately."""
    r = get_redis()
    r.delete(_k(scope, email), _k(f"{scope}_cooldown", email))
    if sends_this_hour(email, scope) > 0:
        r.decr(_k(f"{scope}_sends", email))


# ---------------------------------------------------------------- verification
class OtpResult(str, Enum):
    OK = "ok"
    INVALID = "invalid"
    EXPIRED = "expired"            # no OTP pending (expired, never sent, or already used)
    TOO_MANY_ATTEMPTS = "too_many"


def check_otp(email: str, otp: str, scope: str = SIGNUP) -> tuple[OtpResult, str | None, int]:
    """Returns (result, role, attempts_left). The OTP is deleted on success or when attempts run out."""
    s = get_settings()
    r = get_redis()
    key = _k(scope, email)
    data = r.hgetall(key)
    if not data:
        return OtpResult.EXPIRED, None, 0

    attempts = r.hincrby(key, "attempts", 1)
    if attempts > s.otp_max_attempts:
        r.delete(key)
        return OtpResult.TOO_MANY_ATTEMPTS, None, 0

    if otp_matches(email, otp, data["code_hash"]):
        r.delete(key)  # single use
        return OtpResult.OK, data["role"], 0

    left = s.otp_max_attempts - attempts
    if left <= 0:
        r.delete(key)
        return OtpResult.TOO_MANY_ATTEMPTS, None, 0
    return OtpResult.INVALID, None, left


# ---------------------------------------------------------------- single-use signup tokens
def register_signup_token(jti: str, email: str, ttl_seconds: int) -> None:
    get_redis().set(_k("signup_jti", jti), email, ex=ttl_seconds)


def signup_token_is_live(jti: str) -> bool:
    return bool(get_redis().exists(_k("signup_jti", jti)))


def consume_signup_token(jti: str) -> None:
    get_redis().delete(_k("signup_jti", jti))


# ---------------------------------------------------------------- login throttling
def login_locked_seconds(email: str) -> int:
    s = get_settings()
    r = get_redis()
    fails = r.get(_k("login_fail", email))
    if fails and int(fails) >= s.login_max_failures:
        return max(r.ttl(_k("login_fail", email)), 1)
    return 0


def record_login_failure(email: str) -> None:
    _incr_with_window(_k("login_fail", email), get_settings().login_failure_window_seconds)


def clear_login_failures(email: str) -> None:
    get_redis().delete(_k("login_fail", email))
