"""
Test JWT authentication:
  - login with correct/incorrect credentials
  - /auth/me with and without a valid token
  - protected endpoints (query/compare/build-index) reject anonymous calls
  - health stays public

Requires the seed admin to exist in db/analytics.db (run db/setup_sqlite.py)
and ADMIN_PASSWORD in the environment to match.
"""
import os

from fastapi.testclient import TestClient

from api.app import app

ADMIN = os.getenv("ADMIN_USERNAME", "admin")
PW = os.getenv("ADMIN_PASSWORD", "change-me")


def main() -> None:
    with TestClient(app) as c:
        # Wrong password is rejected.
        assert c.post("/api/v1/auth/login", json={"username": ADMIN, "password": "nope"}).status_code == 401

        # Correct credentials yield a bearer token.
        r = c.post("/api/v1/auth/login", json={"username": ADMIN, "password": PW})
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        assert r.json()["token_type"] == "bearer"
        headers = {"Authorization": f"Bearer {token}"}

        # /auth/me requires a valid token.
        assert c.get("/api/v1/auth/me").status_code == 401
        me = c.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200 and me.json()["username"] == ADMIN

        # Protected endpoints reject anonymous calls...
        assert c.post("/api/v1/query", json={"query": "x"}).status_code == 401
        assert c.post("/api/v1/compare", json={"query": "x", "mode": 1}).status_code == 401
        assert c.post("/api/v1/build-index").status_code == 401

        # ...and accept an authenticated one (build-index returns immediately).
        assert c.post("/api/v1/build-index", headers=headers).status_code == 200

        # Health is public; garbage tokens are rejected.
        assert c.get("/api/v1/health").status_code == 200
        assert c.get("/api/v1/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401

    print("All auth assertions passed.")


if __name__ == "__main__":
    main()
