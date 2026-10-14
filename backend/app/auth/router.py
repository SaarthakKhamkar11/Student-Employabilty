"""Auth endpoints.

Signup (new users):
  1. POST /auth/signup/request-otp   {email, role}            -> OTP emailed
  2. POST /auth/signup/verify-otp    {email, otp}             -> signup_token (15 min, single use)
  3. POST /auth/signup/complete      {signup_token, password, full_name, *_profile}
                                                              -> account created + access_token
Login (existing users, no OTP):
     POST /auth/login                {email, password}        -> access_token
     GET  /auth/me                   Authorization: Bearer <access_token>
"""
import jwt
import psycopg
from fastapi import APIRouter, Depends, HTTPException, status

from . import otp_store
from .config import get_settings
from .db import get_conn
from .deps import get_current_user
from .email_sender import EmailSendError, send_otp_email
from .otp_store import OtpResult
from .schemas import (LoginIn, OtpRequestIn, OtpRequestOut, OtpVerifyIn, SignupCompleteIn,
                      SignupTokenOut, TokenOut, UserOut)
from .security import (PURPOSE_SIGNUP, burn_password_check, create_access_token,
                       create_signup_token, decode_token, generate_otp, hash_otp, hash_password,
                       verify_password)

router = APIRouter(prefix="/auth", tags=["auth"])


def _email_registered(email: str) -> bool:
    with get_conn() as conn:
        return conn.execute("SELECT 1 FROM app_users WHERE lower(email) = %s", (email,)).fetchone() is not None


def _too_many(detail: str, retry_after: int) -> HTTPException:
    return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=detail,
                         headers={"Retry-After": str(retry_after)})


# ================================================================ signup step 1
@router.post("/signup/request-otp", response_model=OtpRequestOut)
def request_signup_otp(body: OtpRequestIn):
    s = get_settings()
    email = body.email

    if _email_registered(email):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="An account with this email already exists. Please log in.")

    wait = otp_store.resend_wait_seconds(email)
    if wait > 0:
        raise _too_many(f"Please wait {wait} seconds before requesting another code.", wait)
    if otp_store.sends_this_hour(email) >= s.otp_max_sends_per_hour:
        raise _too_many("Too many codes requested for this email. Try again in an hour.", 3600)

    otp = generate_otp()
    otp_store.save_otp(email, hash_otp(email, otp), body.role)
    try:
        send_otp_email(email, otp)
    except EmailSendError:
        otp_store.discard_otp(email)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY,
                            detail="Couldn't send the verification email. Please try again.")

    return OtpRequestOut(message=f"Verification code sent to {email}",
                         expires_in=s.otp_expiry_seconds,
                         resend_after=s.otp_resend_cooldown_seconds)


# ================================================================ signup step 2
@router.post("/signup/verify-otp", response_model=SignupTokenOut)
def verify_signup_otp(body: OtpVerifyIn):
    result, role, left = otp_store.check_otp(body.email, body.otp)

    if result is OtpResult.EXPIRED:
        raise HTTPException(status.HTTP_410_GONE, detail="Code expired or not found. Request a new one.")
    if result is OtpResult.TOO_MANY_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Too many wrong attempts. Request a new code.")
    if result is OtpResult.INVALID:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=f"Incorrect code. {left} attempt(s) left.")

    token, jti, expires_in = create_signup_token(body.email, role)
    otp_store.register_signup_token(jti, body.email, expires_in)
    return SignupTokenOut(signup_token=token, role=role, expires_in=expires_in)


# ================================================================ signup step 3
@router.post("/signup/complete", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def complete_signup(body: SignupCompleteIn):
    try:
        claims = decode_token(body.signup_token, PURPOSE_SIGNUP)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="Signup session expired. Please verify your email again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid signup token.")

    if not otp_store.signup_token_is_live(claims["jti"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="This signup link was already used. Please log in.")

    email, role = claims["sub"], claims["role"]
    if role == "student" and body.student_profile is None:
        raise HTTPException(422, detail="student_profile is required")
    if role == "recruiter" and body.recruiter_profile is None:
        raise HTTPException(422, detail="recruiter_profile is required")

    pw_hash = hash_password(body.password)
    try:
        with get_conn() as conn:  # one transaction: user + profile together or not at all
            user_id = conn.execute(
                "INSERT INTO app_users (email, password_hash, role, full_name) "
                "VALUES (%s, %s, %s, %s) RETURNING user_id",
                (email, pw_hash, role, body.full_name),
            ).fetchone()["user_id"]

            if role == "student":
                p = body.student_profile
                conn.execute(
                    "INSERT INTO student_accounts (user_id, degree, branch, graduation_year, gender) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (user_id, p.degree.strip(), p.branch.strip(), p.graduation_year, p.gender),
                )
            else:
                p = body.recruiter_profile
                conn.execute(
                    "INSERT INTO recruiter_accounts (user_id, company_name, designation) VALUES (%s, %s, %s)",
                    (user_id, p.company_name.strip(), p.designation.strip() if p.designation else None),
                )
    except psycopg.errors.UniqueViolation:
        otp_store.consume_signup_token(claims["jti"])
        raise HTTPException(status.HTTP_409_CONFLICT,
                            detail="An account with this email already exists. Please log in.")

    otp_store.consume_signup_token(claims["jti"])
    token, expires_in = create_access_token(user_id, role)
    return TokenOut(access_token=token, role=role, expires_in=expires_in)


# ================================================================ login
@router.post("/login", response_model=TokenOut)
def login(body: LoginIn):
    email = body.email
    locked = otp_store.login_locked_seconds(email)
    if locked:
        raise _too_many(f"Too many failed attempts. Try again in {max(locked // 60, 1)} minute(s).", locked)

    with get_conn() as conn:
        user = conn.execute(
            "SELECT user_id, password_hash, role, is_active FROM app_users WHERE lower(email) = %s",
            (email,),
        ).fetchone()

    if user is None:
        burn_password_check(body.password)
        otp_store.record_login_failure(email)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if not verify_password(body.password, user["password_hash"]):
        otp_store.record_login_failure(email)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if not user["is_active"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This account has been disabled")

    otp_store.clear_login_failures(email)
    with get_conn() as conn:
        conn.execute("UPDATE app_users SET last_login_at = now() WHERE user_id = %s", (user["user_id"],))

    token, expires_in = create_access_token(user["user_id"], user["role"])
    return TokenOut(access_token=token, role=user["role"], expires_in=expires_in)


# ================================================================ current user
@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    table = "student_accounts" if user["role"] == "student" else "recruiter_accounts"
    with get_conn() as conn:
        profile = conn.execute(f"SELECT * FROM {table} WHERE user_id = %s", (user["user_id"],)).fetchone() or {}
    profile.pop("user_id", None)
    return UserOut(**user, profile=profile)
