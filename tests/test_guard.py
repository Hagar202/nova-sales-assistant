"""The fact guard: made-up prices and percentages are caught, real ones and simple math pass."""

import pytest

from app.guard import money_in


@pytest.fixture()
def guard(assistant):
    return assistant.guard


@pytest.fixture()
def context(assistant):
    return lambda company: assistant.lead_docs[assistant.find_lead(company)].page_content


def test_reads_money_formats():
    values = [v for _, v in money_in("It costs $25,000, or $1.5k, or 4000 USD.")]
    assert values == [25000, 1500, 4000]
    assert money_in("They pay $1,250, then more.")[0][0] == "$1,250"  # no trailing comma in warnings


CORRECT_ANSWERS = [
    ("Nile Pharma", "The Advanced App costs $25,000 and is delivered in 16 weeks."),
    ("Nile Pharma", "With 10% off, the Advanced App is $22,500."),
    ("Nile Pharma", "After 10% off ($22,500): $11,250 / $6,750 / $4,500."),
    ("Nile Pharma", "Payments: $12,500 at signing, $7,500 in the middle and $5,000 at delivery."),
    ("Nile Pharma", "The chatbot first year: $5,000 + 12 × $300 = $8,600."),
    ("Nile Pharma", "The AI Chatbot costs $5,000 + $300 / month."),
    ("Nile Pharma", "Ads for 3 months: $3,600 + $1,500 ad budget = $5,100."),
    ("Nile Pharma", "Ads Management costs $1,200 per month plus at least $500 ad budget."),
    ("Nile Pharma", "Reach 120% of the $30,000 target, i.e. $36,000, to get the $300 bonus."),
    ("Nile Pharma", "Commission is 5% on apps, so $1,250 on this deal."),
    ("Nile Pharma", "With a 15% bundle discount, $25,000 becomes $21,250."),
    ("Nile Pharma", "Maintenance after 3 months is $200 per month; extra changes $40 per hour."),
    ("Nile Pharma", "SEO costs $1,000/month with a 6-month minimum, so $6,000."),
    ("Nile Pharma", "Up to 10% without approval, max 20% with manager approval."),
    ("Nile Pharma", "Suggest the Standard App ($12,000) or the Basic App ($6,000)."),
    ("Elite Clinics", "10% off for 12 months: $720/month, $8,640 for the year."),
    ("Cairo Logistics", "20% off $10,000 = $8,000; payments $4,000, $2,400, $1,600."),
    ("Sinai Tours", "Their budget of $4,000 is $21,000 short of the Advanced App ($25,000)."),
    ("Sinai Tours", "The Basic App ($6,000) is still above their $4,000 budget."),
    ("Sinai Tours", "They want the Advanced App, but their budget is $4,000."),
    ("Nile Pharma", "I don't have this information."),
]


@pytest.mark.parametrize("company,answer", CORRECT_ANSWERS)
def test_correct_answers_pass(guard, context, company, answer):
    assert guard.unsupported(answer, context(company)) == []


@pytest.mark.parametrize("answer,flag", [
    ("The Advanced App costs $18,000.", "$18,000"),
    ("The Basic App costs $5,000.", "$5,000"),            # a real price, but of another package
    ("The Standard App is $13,000.", "$13,000"),
    ("The E-commerce Store is $9,000.", "$9,000"),
    ("You can give 25% off without approval.", "25%"),
    ("Commission is 8% on apps.", "8%"),
])
def test_made_up_numbers_are_caught(guard, context, answer, flag):
    problems = guard.unsupported(answer, context("Nile Pharma"))
    assert problems and problems[0].startswith(flag)


def test_numbers_the_rep_wrote_count_as_known(guard, context):
    assert guard.unsupported("Yes, $3,000 is below the price.", context("Nile Pharma"),
                             user_text="They can pay 3000 dollars") == []
