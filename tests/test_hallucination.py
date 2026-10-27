"""The assistant catches made-up prices: it asks the model again, and flags what is still wrong."""

from app.assistant import NO_INFO, SalesAssistant
from app.embeddings import HashEmbeddings


def make(leads, chunks, answers):
    replies = iter(answers)
    seen = []

    def fake(messages, max_new_tokens=300):
        seen.append(messages)
        return next(replies)

    return SalesAssistant(leads, chunks, HashEmbeddings(), fake), seen


def test_made_up_price_is_fixed_by_retry(leads, chunks):
    assistant, seen = make(leads, chunks, ["The Advanced App costs $18,000.", "The Advanced App costs $25,000."])
    result = assistant.answer("How much is the Advanced App for Nile Pharma?", [], "")
    assert result["answer"] == "The Advanced App costs $25,000."
    assert result["warnings"] == []
    assert "$18,000" in seen[1][-1]["content"]  # the model was told which number was wrong


def test_price_still_wrong_after_retry_is_flagged(leads, chunks):
    assistant, _ = make(leads, chunks, ["The Advanced App costs $18,000.", "The Advanced App costs $18,000."])
    result = assistant.answer("How much is the Advanced App for Nile Pharma?", [], "")
    assert len(result["warnings"]) == 1 and "$18,000" in result["warnings"][0]


def test_retry_that_gives_up_does_not_hide_the_first_answer(leads, chunks):
    first = "Nile Pharma's Advanced App is $25,000, and plus a setup fee of $12,345."
    assistant, _ = make(leads, chunks, [first, NO_INFO])
    result = assistant.answer("What would Nile Pharma pay?", [], "")
    assert result["answer"] == first and result["warnings"]


def test_correct_answer_needs_no_retry(leads, chunks):
    assistant, seen = make(leads, chunks, ["Nile Pharma has a budget of $25,000, enough for the Advanced App."])
    result = assistant.answer("Is Nile Pharma's budget enough?", [], "")
    assert len(seen) == 1 and result["warnings"] == []


def test_empty_model_answer_becomes_no_info(leads, chunks):
    assistant, _ = make(leads, chunks, [""])
    assert assistant.answer("What is the capital of France?", [], "")["answer"] == NO_INFO


def test_system_prompt_forbids_outside_knowledge(assistant):
    messages = assistant.build_messages("What is the capital of France?", [], "", "context")
    assert "ONLY the Reference data" in messages[0]["content"]
    assert NO_INFO in messages[0]["content"]


def test_never_agrees_that_the_price_is_too_high():
    from app.assistant import clean_answer
    question = "They said the price is too high. How should I respond?"
    assert clean_answer(question, "Yes, the price is too high. Suggest the Standard App.") == "Suggest the Standard App."
    assert clean_answer(question, "The price is high for them, but X.") == "The price is high for them, but X."
    assert clean_answer("Is their budget enough?", "Yes, it is.") == "Yes, it is."
    assert clean_answer("What should I say?", "No, say X.") == "Say X."
