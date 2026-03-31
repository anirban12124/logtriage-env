import uuid
import time
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import LogTriageEnv
from tasks import TASKS

app = FastAPI(title="LogTriage Environment", version="3.0")


# ─── Session Management ─────────────────────────────────────────

class SessionEntry:
    def __init__(self, env: LogTriageEnv, task_id: str):
        self.env = env
        self.task_id = task_id
        self.created_at = time.time()
        self.last_accessed = time.time()


sessions: Dict[str, SessionEntry] = {}
MAX_SESSIONS = 10
TTL_SECONDS = 1800  # 30 minutes


def _cleanup_expired():
    now = time.time()
    expired = [sid for sid, s in sessions.items()
               if now - s.last_accessed > TTL_SECONDS]
    for sid in expired:
        del sessions[sid]


def _get_session(session_id: str) -> SessionEntry:
    _cleanup_expired()
    if session_id not in sessions:
        raise HTTPException(status_code=404,
                            detail="Session not found. Call /reset first.")
    entry = sessions[session_id]
    entry.last_accessed = time.time()
    return entry


# ─── Request Models ──────────