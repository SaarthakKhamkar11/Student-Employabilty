"""Auth settings, read from environment variables / .env (names are case-insensitive)."""
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"

    # Infrastructure (already in your .env)
    database_url: str
    redis_url: str = "redis://redis:6379/0"

    # JWT
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24      # login session: 1 day
    signup_token_expire_minutes: int = 15           # time to finish password + profile after OTP

    # Gmail SMTP (leave SMTP_USER empty in development to print OTPs to the backend log instead)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from_name: str = "Employability Platform"

    # OTP rules
    otp_expiry_seconds: int = 600          # OTP valid for 10 minutes
    otp_max_attempts: int = 5              # wrong guesses before the OTP is burned
    otp_resend_cooldown_seconds: int = 60  # wait before "resend OTP"
    otp_max_sends_per_hour: int = 5        # per email address

    # Login brute-force protection
    login_max_failures: int = 10           # per email, within the window below
    login_failure_window_seconds: int = 900

    @model_validator(mode="after")
    def _check_secret(self):
        weak = len(self.jwt_secret) < 32 or self.jwt_secret in {"test-secret", "secret", "changeme"}
        if weak and self.environment.lower() != "development":
            raise ValueError("JWT_SECRET is too weak for a non-development environment. "
                             "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\"")
        return self

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_user and self.smtp_password)


@lru_cache
def get_settings() -> AuthSettings:
    return AuthSettings()
