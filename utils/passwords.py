"""
utils/passwords.py — Password hashing helpers (bcrypt).

Kept in utils/ (not api/auth) so both the DB seed script (db/setup_sqlite.py)
and the API auth layer can hash/verify with identical logic without creating a
db -> api import dependency.
"""

import bcrypt

# bcrypt truncates silently at 72 bytes; we hash within that limit explicitly.
_MAX_BCRYPT_BYTES = 72


def _truncate(password: str) -> bytes:
    return password.encode("utf-8")[:_MAX_BCRYPT_BYTES]


def hash_password(password: str) -> str:
    """Return a bcrypt hash (utf-8 string) for the given plaintext password."""
    return bcrypt.hashpw(_truncate(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True if the plaintext password matches the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(_truncate(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False
