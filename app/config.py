"""Settings read from environment variables (or a .env file), so nothing is hard-coded for one machine."""

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# The CRM sheet is a snapshot exported on this date. "At risk" and "days since contact"
# are counted from it, so the demo looks the same any day. Set AS_OF_DATE=today for live data.
DATA_SNAPSHOT_DATE = "2026-10-22"


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv(path=ENV_FILE):
    """Read KEY=VALUE lines from .env without overriding variables that are already set."""
    file = Path(path)
    if os.getenv("NOVA_SKIP_DOTENV") or not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int(name, default):
    value = os.getenv(name, "").strip()
    try:
        return int(value) if value else default
    except ValueError as error:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from error


def _date(name, default):
    value = os.getenv(name, default).strip()
    if value.lower() == "today":
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be YYYY-MM-DD or 'today', got {value!r}") from error


@dataclass(frozen=True)
class Settings:
    leads_path: str
    pdf_path: str
    llm_backend: str            # transformers | openai | mock
    llm_model: str              # Hugging Face model for the transformers backend
    openai_base_url: str        # for the openai backend (Ollama, vLLM, Groq...)
    openai_model: str
    openai_api_key: str
    llm_timeout_seconds: int
    embedding_model: str
    mock_llm: bool              # True = no model download, for UI development and tests
    as_of_date: date            # "today" for the CRM rules (days since contact, at risk)
    session_ttl_minutes: int    # a chat is forgotten after this many idle minutes
    max_sessions: int
    max_message_chars: int

    @property
    def model_label(self):
        if self.mock_llm or self.llm_backend == "mock":
            return "mock"
        return self.openai_model if self.llm_backend == "openai" else self.llm_model


def get_settings():
    _load_dotenv()
    mock = _bool("MOCK_LLM", False)
    backend = os.getenv("LLM_BACKEND", "transformers").strip().lower()
    return Settings(
        leads_path=os.getenv("LEADS_PATH", "data/Nova_Digital_Leads.xlsx"),
        pdf_path=os.getenv("PDF_PATH", "data/Nova_Digital_Sales_Playbook.pdf"),
        llm_backend="mock" if mock else backend,
        llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
        openai_model=os.getenv("OPENAI_MODEL", "qwen2.5:7b-instruct"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 180),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        mock_llm=mock,
        as_of_date=_date("AS_OF_DATE", DATA_SNAPSHOT_DATE),
        session_ttl_minutes=_int("SESSION_TTL_MINUTES", 120),
        max_sessions=_int("MAX_SESSIONS", 500),
        max_message_chars=_int("MAX_MESSAGE_CHARS", 1000),
    )
