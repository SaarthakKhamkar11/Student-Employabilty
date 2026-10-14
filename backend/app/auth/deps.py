"""FastAPI dependencies to protect routes.

    from .auth import get_current_user, require_role

    @app.get("/students/me/matches")
    def my_matches(user = Depends(require_role("student"))): ...

    @app.post("/jobs")
    def post_job(user = Depends(require_role("recruiter"))): ...
"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .db import get_conn
from .security import PURPOSE_ACCESS, decode_token

_bearer = HTTPBearer(auto_error=False)


def _unauthorised(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
    """Returns {user_id, email, role, full_name} for a valid, active user, else 401."""
    if creds is None:
        raise _unauthorised("Not logged in")
    try:
        payload = decode_token(creds.credentials, PURPOSE_ACCESS)
    except jwt.ExpiredSignatureError:
        raise _unauthorised("Session expired, please log in again")
    except jwt.InvalidTokenError:
        raise _unauthorised("Invalid token")

    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError):
        raise _unauthorised("Invalid token")

    with get_conn() as conn:
        user = conn.execute(
            "SELECT user_id, email, role, full_name, is_active FROM app_users WHERE user_id = %s",
            (user_id,),
        ).fetchone()
    if not user or not user["is_active"]:
        raise _unauthorised("Account not found or disabled")
    user.pop("is_active")
    return user


def require_role(*roles: str):
    def _dep(user: dict = Depends(get_current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not allowed for your account type")
        return user
    return _dep
