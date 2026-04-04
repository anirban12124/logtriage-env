"""
Root-level app re-export for uvicorn.

The Dockerfile CMD runs: uvicorn app:app --host 0.0.0.0 --port 8000
This module re-exports the FastAPI app from server/app.py so the CMD resolves.
"""
from server.app import app  # noqa: F401
