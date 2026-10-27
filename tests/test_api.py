"""API tests with FastAPI's TestClient (no server, no GPU)."""

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


@pytest.fixture()
def client(assistant):
    with TestClient(create_app(assistant=assistant, settings=get_settings())) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["leads"] == 200 and body["chunks"] == 229


def test_list_and_filter_leads(client):
    assert len(client.get("/api/leads").json()) == 200
    hot = client.get("/api/leads", params={"priority": "hot"}).json()
    assert hot and all(lead["priority"] == "Hot" for lead in hot)
    assert client.get("/api/leads", params={"search": "nile"}).json()[0]["lead_id"] == "L001"


def test_lead_details_and_404(client):
    body = client.get("/api/leads/L001").json()
    assert body["company"] == "Nile Pharma" and body["check"]["priority"] == "Hot"
    assert client.get("/api/leads/L999").status_code == 404


def test_chat_session_memory(client):
    first = client.post("/api/chat", json={"message": "Tell me about Nile Pharma."}).json()
    session = first["session_id"]
    second = client.post("/api/chat", json={"message": "What do they want?", "session_id": session}).json()
    assert second["session_id"] == session
    assert second["lead_id"] == "L001"
    assert second["memory"]["turns_in_full"] == 2


def test_sessions_are_separate(client):
    a = client.post("/api/chat", json={"message": "Tell me about Nile Pharma."}).json()
    b = client.post("/api/chat", json={"message": "Payment terms?"}).json()
    assert a["session_id"] != b["session_id"]
    assert b["lead_id"] is None


def test_clear_session(client):
    session = client.post("/api/chat", json={"message": "Hi"}).json()["session_id"]
    assert client.delete(f"/api/sessions/{session}").status_code == 204
    assert client.get(f"/api/sessions/{session}/memory").json()["turns_in_full"] == 0


def test_bad_requests(client):
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "lead_id": "L999"}).status_code == 404


def test_web_app_is_served(client):
    assert "Nova Sales Assistant" in client.get("/").text


def test_health_shows_backend_and_date(client):
    body = client.get("/api/health").json()
    assert body["as_of"] == "2026-10-22"
    assert body["backend"] in ("transformers", "openai", "mock")


def test_chat_returns_sources(client):
    body = client.post("/api/chat", json={"message": "Tell me about Nile Pharma."}).json()
    assert body["lead_ids"] == ["L001"]
    assert body["sources"][0] == {"kind": "lead", "label": "CRM: Nile Pharma (L001)"}
    assert body["warnings"] == []


def test_lead_id_is_case_insensitive(client):
    assert client.get("/api/leads/l001").json()["lead_id"] == "L001"


def test_at_risk_filter(client):
    risky = client.get("/api/leads", params={"at_risk": True}).json()
    assert risky and all(lead["at_risk"] for lead in risky)


def test_bad_session_id_is_rejected(client):
    assert client.post("/api/chat", json={"message": "hi", "session_id": "../../etc"}).status_code == 422


def test_model_error_gives_502_and_keeps_memory(leads, chunks):
    from app.assistant import SalesAssistant
    from app.embeddings import HashEmbeddings
    from app.llm import LLMError

    calls = {"n": 0}

    def flaky(messages, max_new_tokens=300):
        calls["n"] += 1
        if calls["n"] == 2:
            raise LLMError("The model server at http://localhost:11434/v1 is not reachable (ConnectError).")
        return "Nile Pharma wants an Advanced App."

    assistant = SalesAssistant(leads, chunks, HashEmbeddings(), flaky)
    with TestClient(create_app(assistant=assistant, settings=get_settings())) as c:
        session = c.post("/api/chat", json={"message": "Tell me about Nile Pharma."}).json()["session_id"]
        failed = c.post("/api/chat", json={"message": "And their budget?", "session_id": session})
        assert failed.status_code == 502 and "not reachable" in failed.json()["detail"]
        assert c.get(f"/api/sessions/{session}/memory").json()["turns_in_full"] == 1


def test_favicon(client):
    assert client.get("/favicon.ico").status_code == 200
