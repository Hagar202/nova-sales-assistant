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
