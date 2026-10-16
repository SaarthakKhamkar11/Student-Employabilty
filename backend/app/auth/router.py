"""Auth endpoints.

Signup (new users):
  1. POST /auth/signup/request-otp   {email, role}            -> OTP emailed
  2. POST /auth/signup/verify-otp    {email, otp}             -> signup_token (15 min, single use)
  3. POST /auth/signup/complete      {signup_token, password, full_name, *_profile}
                                                              -> account created + access_token

Login (existing users, no OTP):
     POST /auth/login                {email, password, role?} -> access_token
     GET  /auth/me                   Authorization: Bearer <access_token>

Forgot password (existing users):
     POST /auth/password/forgot      {email}                  -> reset code emailed
     POST /auth/password/reset       {email, otp, password}   -> password replaced; log in with it

Accounts live in `users` (migration 0009 + 0012). A student's signup also creates the student's own
`student_profiles` row (degree, branch, graduation year), so the Profile Agent combines those stated facts with the
resume when it is uploaded (app/agents/profile_agent.py: "the stored record supplies the fields a resume does not
state"). A company's signup creates or reuses its `companies` row.
"""
import uuid

import jwt
import psycopg
from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.types.json import Jsonb

from . import otp_store
from .config import get_settings
from .db import get_conn
from .deps import get_current_user
from .email_sender import EmailSendError, send_otp_email
from .otp_store import OtpResult
from .schemas import (LoginIn, MessageOut, OtpRequestIn, OtpRequestOut, OtpVerifyIn, PasswordForgotIn,
                      PasswordResetIn, SignupCompleteIn, SignupTokenOut, TokenOut, UserOut)
from .security import (PURPOSE_SIGNUP, create_access_token,
                       create_signup_token, decode_token, generate_otp, hash_otp, hash_password,
                       verify_password)

router = APIRouter(prefix="/auth", tags=["auth"])

# The student_profiles source for portal accounts. "resume_upload" makes a later request by id reuse the
# profile built from the uploaded resume instead of rebuilding it from the (sparse) stored columns.
PORTAL_STUDENT_SOURCE = "resume_upload"
ROLE_LABEL = {"student": "student", "recruiter": "company", "admin": "admin"}


def _email_registered(email: str) -> bool:
    with get_conn() as conn:
        return conn.execute("SELECT 1 FROM users WHERE lower(email) = %s", (email,)).fetchone() is not None


def _too_many(detail: str, retry_after: int) -> HTTPException:
    return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=detail,
                         headers={"Retry-After": str(retry_after)})


# ================================================================ signup step 1
@router.post("/signup/request-otp", response_model=OtpRequestOut)
def request_signup_otp(body: OtpRequestIn):
    s = get_settings()
    email = body.email
    if _email_registered(email):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email already registered. Please log in.")
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
        with get_conn() as conn:  # one transaction: account + linked record together or not at all
            student_id = company_id = None
            if role == "student":
                p = body.student_profile
                student_id = f"U{uuid.uuid4().hex[:12]}"  # the Profile Agent's id format for uploads
                conn.execute(
                    "INSERT INTO student_profiles (student_id, degree, branch, graduation_year, source_dataset, "
                    "is_synthetic) VALUES (%s, %s, %s, %s, %s, false)",
                    (student_id, p.degree, p.branch, p.graduation_year, PORTAL_STUDENT_SOURCE))
                details = {"college": p.college}
            else:
                p = body.recruiter_profile
                row = conn.execute("SELECT company_id FROM companies WHERE lower(name) = lower(%s)",
                                   (p.company_name,)).fetchone()
                if row is None:
                    row = conn.execute(
                        "INSERT INTO companies (name, industry, company_size, website, source_dataset) "
                        "VALUES (%s, %s, %s, %s, 'portal_signup') RETURNING company_id",
                        (p.company_name, p.industry, p.company_size, p.website)).fetchone()
                company_id = row["company_id"]
                details = {"designation": p.designation} if p.designation else {}
            user_id = conn.execute(
                "INSERT INTO users (email, password_hash, role, full_name, student_id, company_id, details) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING user_id",
                (email, pw_hash, role, body.full_name, student_id, company_id, Jsonb(details)),
            ).fetchone()["user_id"]
    except psycopg.errors.UniqueViolation:
        otp_store.consume_signup_token(claims["jti"])
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email already registered. Please log in.")
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
            "SELECT user_id, password_hash, role, is_active FROM users WHERE lower(email) = %s", (email,),
        ).fetchone()
    if user is None:
        # Product decision (2026-10-15): tell a new user plainly that there is no account, so they sign up instead
        # of retrying. This reveals whether an email is registered; the per-email lockout below still applies.
        otp_store.record_login_failure(email)
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Account not found. Try signing up instead.")
    if not verify_password(body.password, user["password_hash"]):
        otp_store.record_login_failure(email)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Incorrect password. Please try again.")
    if not user["is_active"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="This account has been disabled")
    # Admins may sign in from either side of the toggle; students and companies only from their own.
    if body.role and user["role"] not in (body.role, "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            detail=f"This is a {ROLE_LABEL[user['role']]} account. "
                                   f"Switch the toggle to \"I'm a {ROLE_LABEL[user['role']].title()}\".")
    otp_store.clear_login_failures(email)
    with get_conn() as conn:
        conn.execute("UPDATE users SET last_login_at = now() WHERE user_id = %s", (user["user_id"],))
    token, expires_in = create_access_token(user["user_id"], user["role"])
    return TokenOut(access_token=token, role=user["role"], expires_in=expires_in)


# ================================================================ forgot password
@router.post("/password/forgot", response_model=OtpRequestOut)
def forgot_password(body: PasswordForgotIn):
    s = get_settings()
    email = body.email
    if not _email_registered(email):
        # Same product decision as login: say plainly that there is no account
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Account not found. Try signing up instead.")
    wait = otp_store.resend_wait_seconds(email, otp_store.RESET)
    if wait > 0:
        raise _too_many(f"Please wait {wait} seconds before requesting another code.", wait)
    if otp_store.sends_this_hour(email, otp_store.RESET) >= s.otp_max_sends_per_hour:
        raise _too_many("Too many codes requested for this email. Try again in an hour.", 3600)
    otp = generate_otp()
    otp_store.save_otp(email, hash_otp(email, otp), "reset", otp_store.RESET)
    try:
        send_otp_email(email, otp, purpose="reset")
    except EmailSendError:
        otp_store.discard_otp(email, otp_store.RESET)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="Couldn't send the reset email. Please try again.")
    return OtpRequestOut(message=f"Reset code sent to {email}", expires_in=s.otp_expiry_seconds,
                         resend_after=s.otp_resend_cooldown_seconds)


@router.post("/password/reset", response_model=MessageOut)
def reset_password(body: PasswordResetIn):
    result, _, left = otp_store.check_otp(body.email, body.otp, otp_store.RESET)
    if result is OtpResult.EXPIRED:
        raise HTTPException(status.HTTP_410_GONE, detail="Code expired or not found. Request a new one.")
    if result is OtpResult.TOO_MANY_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many wrong attempts. Request a new code.")
    if result is OtpResult.INVALID:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Incorrect code. {left} attempt(s) left.")
    with get_conn() as conn:
        updated = conn.execute("UPDATE users SET password_hash = %s WHERE lower(email) = %s RETURNING user_id",
                               (hash_password(body.password), body.email)).fetchone()
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Account not found. Try signing up instead.")
    otp_store.clear_login_failures(body.email)  # a verified reset lifts a lockout from earlier wrong guesses
    return MessageOut(message="Password updated. Log in with your new password.")


# ================================================================ current user
@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(get_current_user)):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT u.details, u.profile_confirmed_at, u.student_id, c.company_id, c.name, c.industry, "
            "c.company_size, c.website, sp.profile IS NOT NULL AS resume_uploaded, "
            "(SELECT count(*) FROM jobs j WHERE j.posted_by = u.user_id) AS jobs_posted "
            "FROM users u LEFT JOIN companies c ON c.company_id = u.company_id "
            "LEFT JOIN student_profiles sp ON sp.student_id = u.student_id WHERE u.user_id = %s",
            (user["user_id"],)).fetchone()
    company = None
    if row["company_id"] is not None:
        company = {"company_id": row["company_id"], "name": row["name"], "industry": row["industry"],
                   "company_size": row["company_size"], "website": row["website"]}
    confirmed = row["profile_confirmed_at"]
    return UserOut(
        user_id=str(user["user_id"]), email=user["email"], role=user["role"], full_name=user["full_name"],
        student_id=row["student_id"], company=company, details=row["details"] or {},
        onboarding={"profile_confirmed_at": confirmed.isoformat() if confirmed else None,
                    "resume_uploaded": bool(row["resume_uploaded"]), "jobs_posted": row["jobs_posted"]})
