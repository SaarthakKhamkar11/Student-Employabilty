"""Email-OTP signup + email/password login with JWT, for students and recruiters."""
from .db import init_auth_schema
from .deps import get_current_user, require_role
from .router import router

__all__ = ["router", "get_current_user", "require_role", "init_auth_schema"]
