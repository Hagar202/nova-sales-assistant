"""Shared test setup: real data and rules, offline embeddings and a fake model (no GPU, no downloads)."""

import os
import sys
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["NOVA_SKIP_DOTENV"] = "1"  # a local .env must not change the tests

from app.assistant import SalesAssistant, load_leads, load_playbook_chunks  # noqa: E402
from app.embeddings import HashEmbeddings  # noqa: E402
from app.llm import mock_generate  # noqa: E402

TODAY = date(2026, 10, 22)


@pytest.fixture(scope="session")
def leads():
    return load_leads("data/Nova_Digital_Leads.xlsx")


@pytest.fixture(scope="session")
def chunks():
    return load_playbook_chunks("data/Nova_Digital_Sales_Playbook.pdf")


@pytest.fixture()
def assistant(leads, chunks):
    return SalesAssistant(leads, chunks, HashEmbeddings(size=256), mock_generate, today=TODAY)
