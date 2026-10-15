"""Password hashing, OTP generation/hashing and JWT creation/verification."""
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from .config import get_settings

# ---------------------------------------------------------------- passwords
# bcrypt only looks at the first 72 bytes of a password; schemas.py rejects longer ones.
_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(_BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# Used when an email isn't registered, so a failed login takes the same time either way
# (otherwise response timing would reveal which emails have accounts).
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


def burn_password_check(password: str) -> None:
    verify_password(password, _DUMMY_HASH)


# ---------------------------------------------------------------- OTP
def generate_otp() -> str:
    """6-digit numeric OTP from a cryptographically secure source."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(email: str, otp: str) -> str:
    """Only this keyed hash is stored in Redis, never the OTP itself."""
    key = get_settings().jwt_secret.encode("utf-8")
    return hmac.new(key, f"{email}:{otp}".encode("utf-8"), hashlib.sha256).hexdigest()


def otp_matches(email: str, otp: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_otp(email, otp), stored_hash)


# ---------------------------------------------------------------- JWT
PURPOSE_ACCESS = "access"   # normal logged-in session
PURPOSE_SIGNUP = "signup"   # short-lived proof that the email was verified by OTP


def create_token(subject: str, purpose: str, minutes: int, **claims) -> tuple[str, str]:
    """Returns (token, jti)."""
    s = get_settings()
    now = datetime.now(timezone.utc)
    jti = uuid.uuid4().hex
    payload = {
        "sub": subject,
        "purpose": purpose,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
        "jti": jti,
        **claims,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm), jti


def decode_token(token: str, expected_purpose: str) -> dict:
    """Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError on any problem."""
    s = get_settings()
    payload = jwt.decode(
        token, s.jwt_secret, algorithms=[s.jwt_algorithm],
        options={"require": ["sub", "exp", "iat", "jti", "purpose"]},
    )
    if payload.get("purpose") != expected_purpose:
        # stops a signup token being used as a login token and vice versa
        raise jwt.InvalidTokenError("wrong token purpose")
    return payload


def create_access_token(user_id, role: str) -> tuple[str, int]:
    minutes = get_settings().access_token_expire_minutes
    token, _ = create_token(str(user_id), PURPOSE_ACCESS, minutes, role=role)
    return token, minutes * 60


def create_signup_token(email: str, role: str) -> tuple[str, str, int]:
    minutes = get_settings().signup_token_expire_minutes
    token, jti = create_token(email, PURPOSE_SIGNUP, minutes, role=role)
    return token, jti, minutes * 60
