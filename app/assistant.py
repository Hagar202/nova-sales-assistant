"""
Core logic of the Nova Sales Lead Assistant (RAG + Memory).

This file has no UI code. It:
  1. loads the leads (Excel / Google Sheet) and the sales playbook (PDF),
  2. checks every lead against the playbook rules (budget, decision maker, priority),
  3. builds two FAISS indexes (everything + playbook only),
  4. answers questions with retrieval + conversation memory + summary memory.

The language model is passed in as a function, so the same class works with
Qwen on a GPU, a smaller model on CPU, or a fake model in tests.
"""

from datetime import date, datetime

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Playbook rules (copied from Nova_Digital_Sales_Playbook.pdf)
# ---------------------------------------------------------------------------

# Price of each package in USD. Marketing packages are monthly.
PACKAGE_PRICES = {
    "Basic Website": 1500,
    "Business Website": 4000,
    "E-commerce Store": 8000,
    "Basic App": 6000,
    "Standard App": 12000,
    "Advanced App": 25000,
    "Social Media Management": 800,
    "Ads Management": 1200,
    "SEO": 1000,
    "AI Chatbot": 5000,  # + $300 / month for hosting and updates
    "Business Automation": 7000,
    "CRM + AI Integration": 10000,
}
MONTHLY_PACKAGES = {"Social Media Management", "Ads Management", "SEO"}

# Section 7: decision makers
DECISION_MAKER_WORDS = ("ceo", "owner", "founder", "general manager", "director", "manager")

# Section 8: next action for every stage
NEXT_ACTION = {
    "New": "Call the lead within 24 hours.",
    "Contacted": "Book a discovery meeting within 3 days.",
    "Proposal Sent": "Follow up within 3 days if there is no reply.",
    "Negotiation": "Follow the discount rules and close the deal.",
    "Won": "Hand over to the project manager.",
    "Lost": "Follow up again after 90 days.",
}
ACTIVE_STAGES = {"New", "Contacted", "Proposal Sent", "Negotiation"}
AT_RISK_DAYS = 14

# Section 9: industries with a special discount
SPECIAL_DISCOUNT_INDUSTRIES = {"Education": 20, "Non-profit": 20}


def format_price(package):
    price = PACKAGE_PRICES.get(package)
    if price is None:
        return "unknown"
    if package == "AI Chatbot":
        return "$5,000 + $300 / month"
    if package in MONTHLY_PACKAGES:
        return f"${price:,} / month"
    return f"${price:,}"


def is_decision_maker(job_title):
    title = str(job_title).lower()
    return any(word in title for word in DECISION_MAKER_WORDS)


def check_lead(row, today=None):
    """Apply the playbook rules to one lead. Pure Python, no LLM involved."""
    today = today or date.today()
    package = row["Interested Service"]
    price = PACKAGE_PRICES.get(package)
    budget = int(row["Budget (USD)"])

    budget_ok = price is not None and budget >= price
    decision_maker = is_decision_maker(row["Job Title"])
    qualified = budget_ok and decision_maker  # every lead in the CRM has a clear need in its notes

    stage = row["Stage"]
    if stage in ("Won", "Lost"):
        priority = stage
    elif not qualified:
        priority = "Cold"
    elif stage in ("Proposal Sent", "Negotiation"):
        priority = "Hot"
    else:
        priority = "Warm"

    last_contact = pd.to_datetime(row["Last Contact"]).date()
    days_since = (today - last_contact).days
    at_risk = stage in ACTIVE_STAGES and days_since > AT_RISK_DAYS

    reasons = []
    if not budget_ok:
        reasons.append(f"Budget ${budget:,} is below the {package} price ({format_price(package)}).")
    if not decision_maker:
        reasons.append(f"{row['Job Title']} is not a decision maker. Reach the owner or a manager first.")

    return {
        "price": format_price(package),
        "budget_ok": budget_ok,
        "decision_maker": decision_maker,
        "qualified": qualified,
        "priority": priority,
        "next_action": NEXT_ACTION.get(stage, ""),
        "days_since_contact": days_since,
        "at_risk": at_risk,
        "special_discount": SPECIAL_DISCOUNT_INDUSTRIES.get(row["Industry"]),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_leads(source):
    """Read leads from an .xlsx file, a .csv file or a Google Sheet CSV export link."""
    if str(source).endswith(".xlsx"):
        df = pd.read_excel(source)
    else:
        df = pd.read_csv(source)
    df["Last Contact"] = pd.to_datetime(df["Last Contact"]).dt.strftime("%Y-%m-%d")
    return df


def lead_to_text(row, check):
    """Turn one CRM row into natural sentences (embedding models read sentences better than tables)."""
    text = (
        f"{row['Company']} ({row['Lead ID']}) is a {row['Industry']} company in {row['City']} "
        f"with {row['Employees']} employees. "
        f"They are interested in: {row['Interested Service']}. "
        f"Budget: ${row['Budget (USD)']} ({row['Budget Type']}). "
        f"Contact person: {row['Contact Person']}, {row['Job Title']}. "
        f"Stage: {row['Stage']}. Lead source: {row['Lead Source']}. Last contact: {row['Last Contact']}. "
        f"Notes: {row['Notes']}"
    )
    status = "Qualified" if check["qualified"] else "Not qualified"
    crm_check = (
        f" CRM check: package price {check['price']}; "
        f"budget is {'enough' if check['budget_ok'] else 'NOT enough'}; "
        f"contact is {'a decision maker' if check['decision_maker'] else 'NOT a decision maker'}; "
        f"{status}; priority {check['priority']}; next action: {check['next_action'].rstrip('.')}"
    )
    if check["at_risk"]:
        crm_check += f"; AT RISK: no contact for {check['days_since_contact']} days"
    return text + crm_check


def load_playbook_chunks(pdf_path, chunk_size=500, chunk_overlap=100):
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Sales Assistant inside the CRM of Nova Digital Solutions,
a digital agency that offers websites, mobile apps, digital marketing and AI solutions.
You help the sales team understand their leads before contacting them.

Rules:
1. Answer using ONLY the Context and the conversation history.
2. Never use your own general knowledge, even if you know the answer.
3. Use the conversation history to understand follow-up questions (words like it, they, them).
4. If the user asks about the conversation itself (previous questions, their name, which leads we discussed), answer from the conversation history.
5. If the user greets you or introduces themselves, reply politely.
6. Always answer in clear natural sentences. Never copy the raw data rows.
7. When comparing a budget with a price, compare the numbers carefully. Only suggest packages that cost less than or equal to the budget.
8. If the answer is not in the Context or the history, say: "I don't have this information."
9. Only mention prices, numbers and packages that appear in the Reference data. Never invent prices.
10. Keep answers short (2 to 4 sentences). Do not write emails or long lists unless the user asks for them.
11. Every lead ends with a "CRM check" calculated from the playbook rules. Trust it for budget, decision maker, priority and next action.
12. You are the assistant. The user is a sales rep. Never say that the user's name is your name."""

SUMMARY_PROMPT = """You summarize a chat between a SALES REP (the user) and an AI ASSISTANT.
Write short bullet points. Always keep:
- the sales rep's name, written as "The sales rep's name is ...",
- every lead (company) discussed and the key facts about it,
- the questions the sales rep asked.
Do not invent anything."""


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------

class SalesAssistant:
    """
    RAG + memory.

    history is a list of {"user": ..., "assistant": ...} turns.
    summary is a string. Both live outside the class (one copy per user session),
    so two sales reps using the app at the same time do not share memory.
    """

    def __init__(self, leads_df, pdf_chunks, embeddings, generate_fn,
                 today=None, summary_batch=3, keep_recent=2):
        self.df = leads_df.reset_index(drop=True)
        self.generate = generate_fn
        self.summary_batch = summary_batch
        self.keep_recent = keep_recent

        self.checks = [check_lead(row, today) for _, row in self.df.iterrows()]
        self.lead_docs = [
            Document(page_content=lead_to_text(row, check), metadata={"lead_id": row["Lead ID"]})
            for (_, row), check in zip(self.df.iterrows(), self.checks)
        ]
        self.pdf_chunks = pdf_chunks

        # Two indexes: everything, and the playbook only.
        # When we already know the lead, we search the playbook only,
        # so rows of other companies do not confuse the model.
        self.vector_store = FAISS.from_documents(self.pdf_chunks + self.lead_docs, embeddings)
        self.pdf_store = FAISS.from_documents(self.pdf_chunks, embeddings)

        # Longest names first, so a long name is not matched by a shorter one inside it.
        names = self.df["Company"].str.lower().tolist()
        self._name_order = sorted(range(len(names)), key=lambda i: -len(names[i]))
        self._names = names

    # ---- finding the lead -------------------------------------------------

    def find_lead(self, text):
        """Index of the lead whose company name appears in the text, or None."""
        text = text.lower()
        for i in self._name_order:
            if self._names[i] in text:
                return i
        return None

    def find_lead_in_chat(self, question, history, summary="", lead_hint=None):
        """
        Which lead is the sales rep talking about? In this order:
        the lead named in the question, the lead open in the CRM screen,
        the last lead named in the chat, the last lead named in the summary.
        """
        index = self.find_lead(question)
        if index is not None:
            return index
        if lead_hint is not None:
            return lead_hint
        for turn in reversed(history):
            index = self.find_lead(turn["user"])
            if index is not None:
                return index
        return self.find_lead(summary) if summary else None

    def index_of(self, lead_id):
        """Row index of a Lead ID such as "L001", or None."""
        matches = self.df.index[self.df["Lead ID"] == lead_id]
        return int(matches[0]) if len(matches) else None

    def lead_id_at(self, index):
        return None if index is None else str(self.df.iloc[index]["Lead ID"])

    def lead_summary(self, index):
        row, check = self.df.iloc[index], self.checks[index]
        return {
            "lead_id": row["Lead ID"],
            "company": row["Company"],
            "industry": row["Industry"],
            "service": row["Interested Service"],
            "stage": row["Stage"],
            "priority": check["priority"],
            "at_risk": check["at_risk"],
        }

    def lead_details(self, index):
        row, check = self.df.iloc[index], self.checks[index]
        return {
            **self.lead_summary(index),
            "city": row["City"],
            "employees": int(row["Employees"]),
            "budget": int(row["Budget (USD)"]),
            "budget_type": row["Budget Type"],
            "contact_person": row["Contact Person"],
            "job_title": row["Job Title"],
            "lead_source": row["Lead Source"],
            "last_contact": row["Last Contact"],
            "notes": row["Notes"],
            "check": check,
        }

    # ---- retrieval --------------------------------------------------------

    def retrieve(self, question, history, summary="", lead_hint=None):
        lead_index = self.find_lead_in_chat(question, history, summary, lead_hint)

        # Add the previous question so follow-ups like "how many employees do they have?" still find something.
        search_query = (history[-1]["user"] + " " + question) if history else question

        if lead_index is not None:
            lead = self.lead_docs[lead_index]
            docs = ([lead]
                    + self.pdf_store.similarity_search(lead.page_content, k=4)
                    + self.pdf_store.similarity_search(search_query, k=4))
        else:
            docs = (self.vector_store.similarity_search(question, k=4)
                    + self.vector_store.similarity_search(search_query, k=4))

        # Remove duplicate chunks (the two searches often return the same ones)
        unique, seen = [], set()
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return unique, lead_index

    # ---- prompt -----------------------------------------------------------

    def build_messages(self, question, history, summary, context):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if summary:
            messages.append({"role": "system", "content": "Summary of the earlier conversation:\n" + summary})
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({
            "role": "user",
            "content": "Reference data from the knowledge base (this is NOT our conversation, do not copy it as is):\n"
                       + context + "\n\nMy question: " + question,
        })
        return messages

    # ---- memory -----------------------------------------------------------

    def summarize(self, history, summary):
        """When the history gets long, summarize the oldest turns and keep the most recent ones."""
        if len(history) < self.summary_batch + self.keep_recent:
            return history, summary

        old_text = ""
        for turn in history[:self.summary_batch]:
            old_text += f"Sales rep: {turn['user']}\nAssistant: {turn['assistant']}\n"

        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Old summary:\n{summary}\n\nNew conversation:\n{old_text}"},
        ]
        new_summary = self.generate(messages, max_new_tokens=250)
        return history[self.summary_batch:], new_summary

    # ---- one chat turn ----------------------------------------------------

    def chat(self, question, history, summary="", lead_hint=None):
        """
        One chat turn. lead_hint is the index of the lead open in the UI (optional).
        Returns (answer, new_history, new_summary, lead_index).
        """
        docs, lead_index = self.retrieve(question, history, summary, lead_hint)
        context = "\n\n".join(doc.page_content for doc in docs)
        if lead_index is not None:
            context = f"Lead we are talking about: {self.df.iloc[lead_index]['Company']}\n\n" + context

        messages = self.build_messages(question, history, summary, context)
        answer = self.generate(messages, max_new_tokens=300)

        history = history + [{"user": question, "assistant": answer}]
        history, summary = self.summarize(history, summary)
        return answer, history, summary, lead_index
