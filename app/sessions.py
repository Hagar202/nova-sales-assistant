"""
Chat memory per user session.

Each browser gets a session id. Its memory (recent turns + summary) is kept here,
so two sales reps using the app at the same time never see each other's chat.
Idle sessions expire, and the store has a size limit so memory cannot grow forever.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    history: list = field(default_factory=list)
    summary: str = ""
    last_used: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self, ttl_minutes=120, max_sessions=500):
        self.ttl = ttl_minutes * 60
        self.max_sessions = max_sessions
        self._sessions = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id=None):
        with self._lock:
            self._remove_expired()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
            else:
                if len(self._sessions) >= self.max_sessions:  # drop the least recently used
                    oldest = min(self._sessions, key=lambda key: self._sessions[key].last_used)
                    del self._sessions[oldest]
                session_id = session_id or uuid.uuid4().hex
                session = self._sessions[session_id] = Session()
            session.last_used = time.time()
            return session_id, session

    def get(self, session_id):
        with self._lock:
            self._remove_expired()
            return self._sessions.get(session_id)

    def delete(self, session_id):
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _remove_expired(self):
        now = time.time()
        for key in [k for k, s in self._sessions.items() if now - s.last_used > self.ttl]:
            del self._sessions[key]
