"""The playbook rules are plain Python, so they can be tested exactly."""

from app.assistant import check_lead, format_price, is_decision_maker
from tests.conftest import TODAY


def lead(leads, company):
    return leads[leads["Company"] == company].iloc[0]


def test_data_loads(leads, chunks):
    assert len(leads) == 200
    assert len(chunks) == 29


def test_qualified_lead_in_proposal_is_hot(leads):
    check = check_lead(lead(leads, "Nile Pharma"), TODAY)
    assert check["budget_ok"] and check["decision_maker"] and check["qualified"]
    assert check["priority"] == "Hot"
    assert check["price"] == "$25,000"


def test_small_budget_is_cold(leads):
    check = check_lead(lead(leads, "Sinai Tours"), TODAY)  # $4,000 for a $25,000 app
    assert not check["budget_ok"]
    assert check["priority"] == "Cold"
    assert any("below" in reason for reason in check["reasons"])


def test_intern_is_not_a_decision_maker(leads):
    check = check_lead(lead(leads, "Desert Farms"), TODAY)
    assert not check["decision_maker"]
    assert not check["qualified"]


def test_at_risk_after_14_days(leads):
    check = check_lead(lead(leads, "Sinai Tours"), TODAY)  # last contact 2026-10-05
    assert check["days_since_contact"] == 17
    assert check["at_risk"]


def test_decision_maker_titles():
    for title in ["CEO", "Owner", "Founder", "General Manager", "Sales Director", "Clinic Manager"]:
        assert is_decision_maker(title), title
    for title in ["Intern", "Accountant", "Junior Sales Rep", "Admin Assistant", "IT Specialist"]:
        assert not is_decision_maker(title), title


def test_price_formats():
    assert format_price("SEO") == "$1,000 / month"
    assert format_price("AI Chatbot") == "$5,000 + $300 / month"
    assert format_price("Basic Website") == "$1,500"
