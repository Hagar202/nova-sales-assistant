"""Finding which lead the sales rep means, and the CRM overview."""


def ids(assistant, text):
    return [assistant.lead_id_at(i) for i in assistant.find_leads(text)]


def test_name_case_and_accents_do_not_matter(assistant):
    assert ids(assistant, "tell me about NILE PHARMA") == ["L001"]
    cafe = assistant.lead_id_at(assistant.find_lead("Café Masr"))
    assert cafe is not None and ids(assistant, "what about cafe masr?") == [cafe]


def test_lead_id_in_the_question(assistant):
    assert ids(assistant, "Brief me on l001 please") == ["L001"]


def test_small_typo_is_found(assistant):
    assert ids(assistant, "Tell me about Nile Pharam") == ["L001"]


def test_unrelated_text_finds_nothing(assistant):
    for text in ["What are the payment terms?", "Hi, I'm Hagar.", "What discount can I give?",
                 "They said the price is too high. How should I respond?"]:
        assert ids(assistant, text) == [], text


def test_longer_name_wins_over_shorter_one_inside_it(assistant):
    found = ids(assistant, "Masr Digital Auto Center")
    assert len(found) == 1
    assert assistant.df.iloc[assistant.index_of(found[0])]["Company"] == "Masr Digital Auto Center"


def test_two_leads_in_one_question(assistant):
    assert ids(assistant, "Compare Nile Pharma and Sinai Tours") == ["L001", ids(assistant, "Sinai Tours")[0]]
    docs, leads = assistant.retrieve("Compare Nile Pharma and Sinai Tours", [], "")
    assert len(leads) == 2
    assert sum(1 for d in docs if d.metadata.get("source") == "lead") == 2


def test_follow_up_after_comparison_keeps_both_leads(assistant):
    result = assistant.answer("Compare Nile Pharma and Sinai Tours", [], "")
    follow = assistant.answer("Which one has the bigger budget?", result["history"], result["summary"])
    assert len(follow["lead_indexes"]) == 2


def test_overview_counts_match_the_rules(assistant):
    hot = sum(1 for c in assistant.checks if c["priority"] == "Hot")
    text = assistant.overview_doc.page_content
    assert "200 leads" in text and f"Hot {hot}" in text and "2026-10-22" in text


def test_overview_used_for_pipeline_questions(assistant):
    docs, leads = assistant.retrieve("How many hot leads do we have?", [], "")
    assert leads == [] and docs[0].metadata["source"] == "overview"


def test_sources_are_labelled(assistant):
    result = assistant.answer("Tell me about Nile Pharma.", [], "")
    labels = [s["label"] for s in result["sources"]]
    assert labels[0] == "CRM: Nile Pharma (L001)"
    assert any(label.startswith("Playbook p.") for label in labels)


def test_pipeline_question_ignores_the_open_lead(assistant):
    open_lead = assistant.index_of("L001")
    for question in ["Which lead should I call first?", "What should I prioritize today?",
                     "Which lead has the biggest budget?", "Who should I call today?",
                     "How many hot leads do we have?"]:
        docs, leads = assistant.retrieve(question, [], "", lead_hint=open_lead)
        assert leads == [] and docs[0].metadata["source"] == "overview", question


def test_questions_about_the_open_lead_keep_it(assistant):
    open_lead = assistant.index_of("L001")
    for question in ["How many employees do they have?", "Is their budget enough?", "What is the next step?"]:
        _, leads = assistant.retrieve(question, [], "", lead_hint=open_lead)
        assert leads == [open_lead], question


def test_compare_with_the_open_lead(assistant):
    open_lead = assistant.index_of("L001")
    _, leads = assistant.retrieve("Compare it with Sinai Tours", [], "", lead_hint=open_lead)
    assert leads == [open_lead, assistant.find_lead("Sinai Tours")]
    _, leads = assistant.retrieve("Tell me about Sinai Tours", [], "", lead_hint=open_lead)
    assert leads == [assistant.find_lead("Sinai Tours")]


def test_similar_words_are_not_a_company(assistant):
    assert ids(assistant, "the lead wants a smart building management app") == []


def test_summary_failure_does_not_lose_the_answer(leads, chunks):
    from app.assistant import SalesAssistant
    from app.embeddings import HashEmbeddings

    def model(messages, max_new_tokens=300):
        if "My question:" not in messages[-1]["content"]:
            raise RuntimeError("summary failed")
        return "An answer."

    bot = SalesAssistant(leads, chunks, HashEmbeddings(), model)
    history, summary = [], ""
    for question in ["Hi", "Tell me about Nile Pharma.", "Budget?", "Stage?", "Next step?"]:
        result = bot.answer(question, history, summary)
        history, summary = result["history"], result["summary"]
    assert result["answer"] == "An answer." and len(history) == 5
