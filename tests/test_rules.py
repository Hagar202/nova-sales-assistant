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


def test_cheaper_packages_and_features_in_context(assistant):
    text = assistant.lead_docs[assistant.index_of("L001")].page_content
    assert "Basic App ($6,000, within their budget) includes ONLY: one platform (Android or iPhone), up to 8 screens" in text
    assert "Advanced App includes ONLY:" in text and "bookings" in text


def test_no_cheaper_package_for_smallest_or_marketing():
    from app.assistant import cheaper_packages
    assert cheaper_packages("Basic Website") == []
    assert cheaper_packages("SEO") == []
    assert cheaper_packages("E-commerce Store") == ["Basic Website", "Business Website"]


def test_cheaper_package_above_budget_is_marked(assistant):
    text = assistant.lead_docs[assistant.index_of("L001")].page_content
    sinai = assistant.lead_docs[assistant.find_lead("Sinai Tours")].page_content  # $4,000 budget
    assert "within their budget" in text
    assert "Basic App ($6,000, above their budget)" in sinai


def test_new_referral_lead_gets_4_hour_action(leads):
    rows = leads[(leads["Stage"] == "New") & (leads["Lead Source"] == "Referral")]
    assert len(rows) > 0
    assert "4 hours" in check_lead(rows.iloc[0], TODAY)["next_action"]


def test_budget_text_shows_monthly_or_one_time(assistant):
    texts = [doc.page_content for doc in assistant.lead_docs]
    assert any("per month" in t for t in texts) and any("one-time" in t for t in texts)
    assert "$25,000 one-time" in assistant.lead_docs[assistant.index_of("L001")].page_content


def test_no_negative_days_when_contact_is_after_today(leads):
    from datetime import date
    check = check_lead(lead(leads, "Nile Pharma"), date(2026, 1, 1))
    assert check["days_since_contact"] == 0 and not check["at_risk"]


def test_playbook_chunks_know_their_section(chunks):
    sections = {s for chunk in chunks for s in chunk.metadata["sections"]}
    assert "9. Discount Rules" in sections and "10. Payment Terms" in sections
    assert all(1 <= chunk.metadata["page"] <= 5 for chunk in chunks)
    discount = [c for c in chunks if "without approval" in c.page_content][0]
    assert "9. Discount Rules" in discount.metadata["sections"]
    assert not any("\x7f" in c.page_content for c in chunks)  # PDF bullet characters are cleaned
