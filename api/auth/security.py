"""
api/auth/security.py — JWT issuing/verification and the route guard dependency.

Flow:
  - authenticate_user(): verify username + password against the users table.
  - create_access_token(): sign a short-lived HS256 JWT carrying `sub` (username).
  - get_current_user(): FastAPI dependency that extracts the Bearer token,
    verifies the signature/expiry, and returns the authenticated UserInfo.
    Attach it via Depends() to protect a route. Stateless — no DB hit per request.
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from api.auth.models import UserInfo
from api.auth.users import get_user
from config import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, JWT_SECRET_KEY
from utils.passwords import verify_password

logger = logging.getLogger("api.auth")

# auto_error=False so we can raise a consistent 401 with a WWW-Authenticate header.
_bearer_scheme = HTTPBearer(auto_error=False)

_CREDENTIALS_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def authenticate_user(username: str, password: str) -> UserInfo | None:
    """Return UserInfo if the credentials are valid, else None."""
    user = await get_user(username)
    if not user:
        # Run a dummy verify to keep timing similar whether or not the user exists.
        verify_password(password, "$2b$12$" + "x" * 53)
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return UserInfo(username=user["username"])


def create_access_token(username: str) -> tuple[str, int]:
    """
    Create a signed JWT for `username`.

    Returns (token, expires_in_seconds).
    """
    expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, expires_in


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserInfo:
    """FastAPI dependency: validate the Bearer token and return the current user."""
    if credentials is None or not credentials.credentials:
        raise _CREDENTIALS_EXC
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )
        username = payload.get("sub")
        if not username:
            raise _CREDENTIALS_EXC
    except JWTError as exc:
        logger.info("[auth] Rejected token: %s", exc)
        raise _CREDENTIALS_EXC
    return UserInfo(username=username)
