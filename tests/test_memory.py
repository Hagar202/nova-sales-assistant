"""Follow-up questions and summary memory."""


def test_follow_up_keeps_the_lead(assistant):
    _, history, summary, lead = assistant.chat("Tell me about Nile Pharma.", [], "")
    _, history, summary, lead = assistant.chat("How many employees do they have?", history, summary)
    assert assistant.lead_id_at(lead) == "L001"


def test_lead_open_in_ui_is_used(assistant):
    index = assistant.index_of("L007")
    _, _, _, lead = assistant.chat("Is their budget enough?", [], "", lead_hint=index)
    assert assistant.lead_id_at(lead) == "L007"


def test_lead_named_in_question_wins_over_ui(assistant):
    index = assistant.index_of("L007")
    _, _, _, lead = assistant.chat("Tell me about Nile Pharma.", [], "", lead_hint=index)
    assert assistant.lead_id_at(lead) == "L001"


def test_summary_after_five_turns(assistant):
    history, summary = [], ""
    for question in ["Hi, I'm Hagar.", "Tell me about Nile Pharma.", "What do they want?",
                     "Payment terms?", "Discount rules?"]:
        _, history, summary, _ = assistant.chat(question, history, summary)
    assert len(history) == 2        # the last 2 turns stay in full
    assert summary                  # the oldest 3 were summarized
