# src/session.py
import uuid
import time
from typing import Dict, Optional
from src.environment import LogTriageEnv


class SessionEntry:
    def __init__(self, env: LogTriageEnv, task_id: str):
        self.env = env
        self.task_id = task_id
        self.created_at = time.time()
        self.last_accessed = time.time()


class SessionManager:
    def __init__(self, max_sessions: int = 10, ttl_seconds: int = 1800):
        self.sessions: Dict[str, SessionEntry] = {}
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds

    def cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, s in self.sessions.items()
            if now - s.last_accessed > self.ttl_seconds
        ]
        for sid in expired:
            del self.sessions[sid]

    def create_session(self, task_id: str) -> str:
        self.cleanup_expired()
        if len(self.sessions) >= self.max_sessions:
            raise RuntimeError("Too many active sessions.")

        env = LogTriageEnv()
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionEntry(env, task_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionEntry]:
        self.cleanup_expired()
        entry = self.sessions.get(session_id)
        if entry is None:
            return None
        entry.last_accessed = time.time()
        return entry

    def destroy_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def active_count(self) -> int:
        self.cleanup_expired()
        return len(self.sessions)