"""Sends the OTP email (signup verification or password reset) through Gmail SMTP (App Password)."""
import logging
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

from .config import get_settings

log = logging.getLogger("auth.email")


class EmailSendError(RuntimeError):
    pass


def send_otp_email(to_email: str, otp: str, purpose: str = "signup") -> None:
    s = get_settings()
    minutes = s.otp_expiry_seconds // 60
    reset = purpose == "reset"
    action = "reset your password" if reset else "verify your email"
    unasked = "request a password reset" if reset else "try to sign up"

    if not s.email_enabled:
        if s.environment.lower() == "development":
            # Dev convenience: no SMTP configured -> OTP appears in `docker compose logs backend`
            log.warning("SMTP not configured - DEV %s OTP for %s is %s", purpose, to_email, otp)
            return
        raise EmailSendError("SMTP_USER / SMTP_PASSWORD are not set")

    msg = EmailMessage()
    msg["Subject"] = f"{otp} is your {'password reset' if reset else 'verification'} code"
    msg["From"] = formataddr((s.smtp_from_name, s.smtp_user))
    msg["To"] = to_email
    msg.set_content(
        f"Your {'password reset' if reset else 'verification'} code is {otp}\n\n"
        f"It expires in {minutes} minutes. If you didn't {unasked}, you can ignore this email.\n"
    )
    msg.add_alternative(f"""\
<div style="font-family:Arial,sans-serif;max-width:420px;margin:auto;padding:24px">
  <p style="margin:0 0 12px">Use this code to {action}:</p>
  <p style="font-size:32px;font-weight:bold;letter-spacing:6px;margin:0 0 12px">{otp}</p>
  <p style="color:#666;margin:0">It expires in {minutes} minutes. If you didn't {unasked}, ignore this email.</p>
</div>""", subtype="html")

    try:
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=15) as server:
            server.starttls()
            server.login(s.smtp_user, s.smtp_password)
            server.send_message(msg)
    except (smtplib.SMTPException, OSError) as exc:
        log.exception("Failed to send OTP email to %s", to_email)
        raise EmailSendError(str(exc)) from exc
