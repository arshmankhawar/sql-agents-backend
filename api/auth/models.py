"""api/auth/models.py — Pydantic request/response models for auth."""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds until the access token expires


class UserInfo(BaseModel):
    username: str
