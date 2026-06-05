"""
server.py — Development server entrypoint.

Run with:
    python server.py

Or directly via uvicorn:
    uvicorn api.app:app --reload --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["api", "dag", "planner", "agents", "schema", "blackboard", "db", "utils"],
    )
