"""LLM backends and settings, without any real model."""

import httpx
import pytest

from app.config import get_settings
from app.llm import LLMError, make_openai_generate


def client_answering(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_backend_sends_greedy_request():
    sent = {}

    def handler(request):
        import json
        sent.update(json.loads(request.content))
        sent["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Hello "}}]})

    generate = make_openai_generate("http://localhost:11434/v1/", "qwen2.5:7b-instruct",
                                    client=client_answering(handler))
    assert generate([{"role": "user", "content": "hi"}], max_new_tokens=50) == "Hello"
    assert sent["url"] == "http://localhost:11434/v1/chat/completions"
    assert sent["temperature"] == 0 and sent["max_tokens"] == 50


def test_openai_backend_errors_are_readable():
    generate = make_openai_generate("http://x/v1", "m", client=client_answering(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(LLMError, match="500"):
        generate([{"role": "user", "content": "hi"}])

    def down(request):
        raise httpx.ConnectError("refused")

    generate = make_openai_generate("http://x/v1", "m", client=client_answering(down))
    with pytest.raises(LLMError, match="not reachable"):
        generate([{"role": "user", "content": "hi"}])


def test_settings_defaults(monkeypatch):
    for name in ["MOCK_LLM", "LLM_BACKEND", "AS_OF_DATE"]:
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.llm_backend == "transformers"
    assert settings.as_of_date.isoformat() == "2026-10-22"


def test_mock_setting_wins(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("LLM_BACKEND", "openai")
    assert get_settings().llm_backend == "mock"


def test_bad_date_is_explained(monkeypatch):
    monkeypatch.setenv("AS_OF_DATE", "22/10/2026")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        get_settings()
