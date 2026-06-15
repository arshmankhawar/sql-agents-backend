"""
api/routes/auth.py — Authentication endpoints.

  POST /api/v1/auth/login  — exchange username/password for a JWT access token.
  GET  /api/v1/auth/me     — return the identity in the current bearer token.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth.models import LoginRequest, TokenResponse, UserInfo
from api.auth.security import authenticate_user, create_access_token, get_current_user

logger = logging.getLogger("api.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest) -> TokenResponse:
    user = await authenticate_user(body.username, body.password)
    if user is None:
        logger.info("[auth] Failed login for username=%r", body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user.username)
    logger.info("[auth] Issued token for username=%r", user.username)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me", response_model=UserInfo)
async def me(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    return current_user
