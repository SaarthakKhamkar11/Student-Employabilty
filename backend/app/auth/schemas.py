"""Request/response models for the auth endpoints."""
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, EmailStr, Field, field_validator, model_validator

SignupRole = Literal["student", "recruiter"]  # self-service roles; "admin" is seeded (python -m app.auth.seed_admin)
Role = Literal["student", "recruiter", "admin"]
CompanySize = Literal["1-10", "11-50", "51-200", "201-1000", "1001-5000", "5000+"]


def normalise_email(v: str) -> str:
    return v.strip().lower()


# Emails are stored and compared in lowercase so "Abc@Gmail.com" and "abc@gmail.com" are one account
Email = Annotated[EmailStr, AfterValidator(normalise_email)]


def _clean(v: str | None) -> str | None:
    if v is None:
        return None
    v = " ".join(v.split())
    return v or None


# ---------------------------------------------------------------- signup step 1 & 2
class OtpRequestIn(BaseModel):
    email: Email
    role: SignupRole


class OtpRequestOut(BaseModel):
    message: str
    expires_in: int
    resend_after: int


class OtpVerifyIn(BaseModel):
    email: Email
    otp: str = Field(pattern=r"^\d{6}$", description="6-digit code from the email")


class SignupTokenOut(BaseModel):
    signup_token: str
    role: SignupRole
    expires_in: int


# ---------------------------------------------------------------- signup step 3
class StudentProfileIn(BaseModel):
    college: str = Field(min_length=1, max_length=200)
    degree: str = Field(min_length=1, max_length=100)
    # The branch/major is the strongest matching signal (the active model's top SHAP feature, PROGRESS.md Phase 12), and a
    # resume rarely states it in a parseable way, so it is asked for once here.
    branch: str = Field(min_length=1, max_length=150)
    graduation_year: int

    @field_validator("college", "degree", "branch")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = _clean(v) or ""
        if not v:
            raise ValueError("cannot be blank")
        return v

    @field_validator("graduation_year")
    @classmethod
    def _year(cls, v: int) -> int:
        # validate_profile accepts 1990-2035 (app/tools/profile_tools.py); a signup outside it could never be saved
        if not 1990 <= v <= 2035:
            raise ValueError("graduation year must be between 1990 and 2035")
        return v


class RecruiterProfileIn(BaseModel):
    company_name: str = Field(min_length=1, max_length=200)
    industry: str = Field(min_length=1, max_length=120)
    company_size: CompanySize
    website: str | None = Field(default=None, max_length=300)
    designation: str | None = Field(default=None, max_length=150)

    @field_validator("company_name", "industry", "designation")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        return _clean(v)

    @field_validator("website")
    @classmethod
    def _website(cls, v: str | None) -> str | None:
        v = (v or "").strip()
        if not v:
            return None
        if not v.lower().startswith(("http://", "https://")):
            v = "https://" + v
        if "." not in v.split("://", 1)[1] or " " in v:
            raise ValueError("website must be a URL such as https://example.com")
        return v


class SignupCompleteIn(BaseModel):
    signup_token: str
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=1, max_length=150)
    student_profile: StudentProfileIn | None = None
    recruiter_profile: RecruiterProfileIn | None = None

    @field_validator("full_name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("full_name cannot be blank")
        return v

    @field_validator("password")
    @classmethod
    def _password_rules(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        if not any(c.isalpha() for c in v) or not any(c.isdigit() for c in v):
            raise ValueError("password must contain at least one letter and one number")
        return v

    @model_validator(mode="after")
    def _one_profile(self):
        if self.student_profile and self.recruiter_profile:
            raise ValueError("send either student_profile or recruiter_profile, not both")
        return self


# ---------------------------------------------------------------- login / session
class LoginIn(BaseModel):
    email: Email
    password: str
    # The login page's role toggle. Checked against the account, so a company can't sign in on the student side.
    role: SignupRole | None = None


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    expires_in: int


class CompanyOut(BaseModel):
    company_id: int
    name: str
    industry: str | None
    company_size: str | None
    website: str | None


class OnboardingOut(BaseModel):
    profile_confirmed_at: str | None = None
    resume_uploaded: bool = False
    jobs_posted: int = 0


class UserOut(BaseModel):
    user_id: str
    email: str
    role: Role
    full_name: str
    student_id: str | None = None
    company: CompanyOut | None = None
    details: dict
    onboarding: OnboardingOut
