"""Request/response models for the auth endpoints."""
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, EmailStr, Field, field_validator, model_validator

Role = Literal["student", "recruiter"]
Gender = Literal["male", "female", "other", "prefer_not_to_say"]


def normalise_email(v: str) -> str:
    return v.strip().lower()


# Emails are stored and compared in lowercase so "Abc@Gmail.com" and "abc@gmail.com" are one account
Email = Annotated[EmailStr, AfterValidator(normalise_email)]


# ---------------------------------------------------------------- signup step 1 & 2
class OtpRequestIn(BaseModel):
    email: Email
    role: Role



class OtpRequestOut(BaseModel):
    message: str
    expires_in: int
    resend_after: int


class OtpVerifyIn(BaseModel):
    email: Email
    otp: str = Field(pattern=r"^\d{6}$", description="6-digit code from the email")



class SignupTokenOut(BaseModel):
    signup_token: str
    role: Role
    expires_in: int


# ---------------------------------------------------------------- signup step 3
class StudentProfileIn(BaseModel):
    degree: str = Field(min_length=1, max_length=100)
    branch: str = Field(min_length=1, max_length=150)
    graduation_year: int = Field(ge=2000, le=2100)
    gender: Gender


class RecruiterProfileIn(BaseModel):
    company_name: str = Field(min_length=1, max_length=200)
    designation: str | None = Field(default=None, max_length=150)


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



class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    expires_in: int


class UserOut(BaseModel):
    user_id: int
    email: str
    role: Role
    full_name: str
    profile: dict
