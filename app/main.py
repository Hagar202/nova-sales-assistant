"""
Nova Sales Assistant API.

    uvicorn app.main:app --port 8000

Open http://localhost:8000 for the web app and http://localhost:8000/docs for the API docs.
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .assistant import SalesAssistant, load_leads, load_playbook_chunks
from .config import get_settings
from .embeddings import load_embeddings
from .llm import LLMError, load_generate_fn
from .schemas import ChatRequest, ChatResponse, Health, LeadDetails, LeadSummary, MemoryInfo
from .sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nova")

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"


def _path(value):
    """Data paths work from any folder: relative paths are read from the project folder."""
    path = Path(value)
    return str(path if path.is_absolute() or path.exists() or "://" in value else ROOT_DIR / path)


def build_assistant(settings):
    logger.info("Loading leads and playbook")
    leads = load_leads(_path(settings.leads_path))
    chunks = load_playbook_chunks(_path(settings.pdf_path))
    logger.info("%d leads, %d playbook chunks, CRM date %s", len(leads), len(chunks), settings.as_of_date)

    embeddings = load_embeddings(settings.embedding_model, mock=settings.mock_llm)
    generate = load_generate_fn(settings)
    return SalesAssistant(leads, chunks, embeddings, generate, today=settings.as_of_date)


def create_app(assistant=None, settings=None):
    """App factory. Tests pass their own assistant; in production it is built at startup."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app):
        if app.state.assistant is None:
            app.state.assistant = build_assistant(settings)  # load the model once, not per request
        logger.info("Ready: open http://localhost:8000")
        yield

    app = FastAPI(
        title="Nova Sales Assistant API",
        description="RAG + memory assistant for the Nova Digital Solutions sales team.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.assistant = assistant
    app.state.sessions = SessionStore(settings.session_ttl_minutes, settings.max_sessions)
    # One GPU, one model: answer one question at a time instead of running out of GPU memory
    app.state.generate_lock = threading.Lock()

    def get_assistant():
        if app.state.assistant is None:
            raise HTTPException(503, "The model is still loading. Try again in a minute.")
        return app.state.assistant

    # ---- API ---------------------------------------------------------------

    @app.get("/api/health", response_model=Health, tags=["system"])
    def health():
        assistant = get_assistant()
        return Health(
            status="ok",
            backend=settings.llm_backend,
            model=settings.model_label,
            mock=settings.llm_backend == "mock",
            leads=len(assistant.df),
            chunks=len(assistant.pdf_chunks) + len(assistant.lead_docs),
            as_of=assistant.today.isoformat(),
        )

    @app.get("/api/leads", response_model=list[LeadSummary], tags=["leads"])
    def list_leads(
        priority: str | None = Query(None, description="Hot, Warm, Cold, Won or Lost"),
        search: str | None = Query(None, description="Part of the company name"),
        at_risk: bool | None = Query(None, description="Only leads with no contact for more than 14 days"),
    ):
        assistant = get_assistant()
        leads = [assistant.lead_summary(i) for i in range(len(assistant.df))]
        if priority:
            leads = [lead for lead in leads if lead["priority"].lower() == priority.lower()]
        if search:
            leads = [lead for lead in leads if search.lower() in lead["company"].lower()]
        if at_risk is not None:
            leads = [lead for lead in leads if lead["at_risk"] == at_risk]
        return leads

    @app.get("/api/leads/{lead_id}", response_model=LeadDetails, tags=["leads"])
    def get_lead(lead_id: str):
        assistant = get_assistant()
        index = assistant.index_of(lead_id)
        if index is None:
            raise HTTPException(404, f"No lead with id {lead_id}.")
        return assistant.lead_details(index)

    @app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
    def chat(request: ChatRequest):
        assistant = get_assistant()
        message = request.message.strip()
        if not message:
            raise HTTPException(422, "The message is empty.")
        if len(message) > settings.max_message_chars:
            raise HTTPException(422, f"The message is longer than {settings.max_message_chars} characters.")

        lead_hint = None
        if request.lead_id:
            lead_hint = assistant.index_of(request.lead_id)
            if lead_hint is None:
                raise HTTPException(404, f"No lead with id {request.lead_id}.")

        session_id, session = app.state.sessions.get_or_create(request.session_id)

        started = time.perf_counter()
        with app.state.generate_lock:
            try:
                result = assistant.answer(message, session.history, session.summary, lead_hint=lead_hint)
            except LLMError as error:
                logger.error("Model error: %s", error)
                raise HTTPException(502, str(error)) from error
            except Exception as error:  # never show a stack trace to the user
                logger.exception("Unexpected error while answering")
                raise HTTPException(500, "Something went wrong while answering. Please try again.") from error
            # Memory is only updated when the answer succeeded
            session.history, session.summary = result["history"], result["summary"]
        latency_ms = int((time.perf_counter() - started) * 1000)
        lead_ids = [assistant.lead_id_at(i) for i in result["lead_indexes"]]
        logger.info("chat session=%s leads=%s warnings=%d latency=%dms",
                    session_id[:8], lead_ids, len(result["warnings"]), latency_ms)

        return ChatResponse(
            session_id=session_id,
            answer=result["answer"],
            lead_id=assistant.lead_id_at(result["lead_index"]),
            lead_ids=lead_ids,
            sources=result["sources"],
            warnings=result["warnings"],
            memory=MemoryInfo(turns_in_full=len(session.history), summary=session.summary),
            latency_ms=latency_ms,
        )

    @app.get("/api/sessions/{session_id}/memory", response_model=MemoryInfo, tags=["chat"])
    def get_memory(session_id: str):
        session = app.state.sessions.get(session_id)
        if session is None:
            return MemoryInfo(turns_in_full=0, summary="")
        return MemoryInfo(turns_in_full=len(session.history), summary=session.summary)

    @app.delete("/api/sessions/{session_id}", status_code=204, tags=["chat"])
    def clear_session(session_id: str):
        app.state.sessions.delete(session_id)

    # ---- Web app -------------------------------------------------------------

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    return app


app = create_app()
