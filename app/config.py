"""Settings read from environment variables, so nothing is hard-coded for one machine."""

import os
from dataclasses import dataclass


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    leads_path: str
    pdf_path: str
    llm_model: str
    embedding_model: str
    mock_llm: bool             # True = no model download, for UI development and tests
    session_ttl_minutes: int   # a chat is forgotten after this many idle minutes
    max_sessions: int
    max_message_chars: int


def get_settings():
    return Settings(
        leads_path=os.getenv("LEADS_PATH", "data/Nova_Digital_Leads.xlsx"),
        pdf_path=os.getenv("PDF_PATH", "data/Nova_Digital_Sales_Playbook.pdf"),
        llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        mock_llm=_bool("MOCK_LLM", False),
        session_ttl_minutes=int(os.getenv("SESSION_TTL_MINUTES", "120")),
        max_sessions=int(os.getenv("MAX_SESSIONS", "500")),
        max_message_chars=int(os.getenv("MAX_MESSAGE_CHARS", "1000")),
    )
