"""
Core logic of the Nova Sales Lead Assistant (RAG + Memory).

This file has no web code. It:
  1. loads the leads (Excel / CSV / Google Sheet) and the sales playbook (PDF),
  2. checks every lead against the playbook rules (budget, decision maker, priority),
  3. builds two FAISS indexes (everything + playbook only),
  4. answers questions with retrieval + conversation memory + summary memory,
  5. checks every price and percentage in the answer against the data (see guard.py).

The language model is passed in as a function, so the same class works with
Qwen on a GPU, a model served by Ollama or an API, or a fake model in tests.
"""

import difflib
import logging
import re
import unicodedata
from collections import Counter
from datetime import date

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .guard import FactGuard, money_in, percents_in
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Playbook rules (copied from Nova_Digital_Sales_Playbook.pdf)
# ---------------------------------------------------------------------------

# Price of each package in USD (sections 2 to 5). Marketing packages are monthly.
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

# What each package includes (sections 2 to 5)
PACKAGE_FEATURES = {
    "Basic Website": "up to 5 pages, mobile-friendly design, contact form, 1 year hosting",
    "Business Website": "up to 15 pages, CMS, Arabic and English, basic SEO",
    "E-commerce Store": "online store, payment gateway (cards and Fawry), product management, order tracking",
    "Basic App": "one platform (Android or iPhone), up to 8 screens",
    "Standard App": "Android and iPhone, up to 20 screens, notifications, user accounts",
    "Advanced App": "Android and iPhone, backend server, admin dashboard, online payment, bookings",
    "Social Media Management": "3 platforms, 20 posts per month, monthly report",
    "Ads Management": "Facebook, Instagram and Google ads; ad budget paid separately (minimum $500 per month)",
    "SEO": "keyword research, website optimization, monthly ranking report",
    "AI Chatbot": "website and WhatsApp chatbot, answers 24/7 in Arabic and English",
    "Business Automation": "automating orders, invoices, reports and email follow-ups",
    "CRM + AI Integration": "AI inside the CRM or ERP: smart replies, lead scoring, automatic reports",
}

# Packages that are smaller versions of each other (same service, cheaper first)
PACKAGE_TIERS = [
    ["Basic Website", "Business Website", "E-commerce Store"],
    ["Basic App", "Standard App", "Advanced App"],
]
WEB_AND_APP = {p for tier in PACKAGE_TIERS for p in tier}

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
# Section 14: response time for new leads depends on where they came from
NEW_LEAD_ACTION_BY_SOURCE = {
    "Referral": "Call the lead within 4 hours (referrals are our best leads).",
    "Exhibition": "Contact the lead within 48 hours after the exhibition.",
    "Cold Call": "Call the lead within 24 hours, and start by explaining the problem we solve.",
}
ACTIVE_STAGES = {"New", "Contacted", "Proposal Sent", "Negotiation"}
AT_RISK_DAYS = 14

# Section 9: industries with a special discount
SPECIAL_DISCOUNT_INDUSTRIES = {"Education": 20, "Non-profit": 20}

PRIORITIES = ("Hot", "Warm", "Cold", "Won", "Lost")


def format_price(package):
    price = PACKAGE_PRICES.get(package)
    if price is None:
        return "unknown"
    if package == "AI Chatbot":
        return "$5,000 + $300 / month"
    if package in MONTHLY_PACKAGES:
        return f"${price:,} / month"
    return f"${price:,}"


def format_budget(amount, budget_type):
    return f"${int(amount):,} per month" if str(budget_type).lower() == "monthly" else f"${int(amount):,} one-time"


def is_decision_maker(job_title):
    title = str(job_title).lower()
    return any(word in title for word in DECISION_MAKER_WORDS)


def cheaper_packages(package):
    """Smaller packages of the same service, for the "price is too high" objection."""
    for tier in PACKAGE_TIERS:
        if package in tier:
            return tier[:tier.index(package)]
    return []


def next_action(stage, source):
    if stage == "New" and source in NEW_LEAD_ACTION_BY_SOURCE:
        return NEW_LEAD_ACTION_BY_SOURCE[source]
    return NEXT_ACTION.get(stage, "")


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
    days_since = max(0, (today - last_contact).days)
    at_risk = stage in ACTIVE_STAGES and days_since > AT_RISK_DAYS

    reasons = []
    if price is None:
        reasons.append(f"{package} is not a package in the playbook.")
    elif not budget_ok:
        reasons.append(f"Budget ${budget:,} is below the {package} price ({format_price(package)}).")
    if not decision_maker:
        reasons.append(f"{row['Job Title']} is not a decision maker. Reach the owner or a manager first.")

    return {
        "price": format_price(package),
        "budget_ok": budget_ok,
        "decision_maker": decision_maker,
        "qualified": qualified,
        "priority": priority,
        "next_action": next_action(stage, row.get("Lead Source")),
        "days_since_contact": days_since,
        "at_risk": at_risk,
        "special_discount": SPECIAL_DISCOUNT_INDUSTRIES.get(row["Industry"]),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Lead ID", "Company", "Industry", "City", "Employees", "Interested Service", "Budget (USD)",
    "Budget Type", "Contact Person", "Job Title", "Stage", "Lead Source", "Last Contact", "Notes",
]


def load_leads(source):
    """Read leads from an .xlsx file, a .csv file or a Google Sheet CSV export link."""
    df = pd.read_excel(source) if str(source).lower().endswith((".xlsx", ".xls")) else pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The leads file is missing these columns: {', '.join(missing)}")
    df = df.dropna(subset=["Lead ID", "Company"]).copy()
    for column in REQUIRED_COLUMNS:
        if column not in ("Employees", "Budget (USD)", "Last Contact"):
            df[column] = df[column].fillna("").astype(str).str.strip()
    df["Employees"] = pd.to_numeric(df["Employees"], errors="coerce").fillna(0).astype(int)
    df["Budget (USD)"] = pd.to_numeric(df["Budget (USD)"], errors="coerce").fillna(0).astype(int)
    df["Last Contact"] = pd.to_datetime(df["Last Contact"]).dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


_HEADING = re.compile(r"(?m)^(\d{1,2})\. ([A-Z][^\n]+)$")


def load_playbook_chunks(pdf_path, chunk_size=500, chunk_overlap=100):
    """Split the playbook into chunks. Each chunk knows its page and playbook section."""
    # The PDF bullets come out as a control character; turn them into dashes.
    pages = [(page.extract_text() or "").replace("\x7f", "-") for page in PdfReader(pdf_path).pages]

    headings = []  # (page, offset, "9. Discount Rules")
    for page_number, text in enumerate(pages):
        for match in _HEADING.finditer(text):
            headings.append((page_number, match.start(), f"{match.group(1)}. {match.group(2).strip()}"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    documents = [Document(page_content=text, metadata={"source": "playbook", "page": number})
                 for number, text in enumerate(pages)]
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        page, start = chunk.metadata["page"], chunk.metadata.get("start_index", 0)
        # The section the chunk starts in, plus any section that starts inside it
        sections = []
        for h_page, h_offset, title in headings:
            if (h_page, h_offset) <= (page, start):
                sections = [title]
        sections += [f"{m.group(1)}. {m.group(2).strip()}" for m in _HEADING.finditer(chunk.page_content)
                     if f"{m.group(1)}. {m.group(2).strip()}" not in sections]
        chunk.metadata["page"] = page + 1  # people count pages from 1
        chunk.metadata["sections"] = sections or ["Introduction"]
        chunk.metadata["section"] = " / ".join(chunk.metadata["sections"])
    return chunks


def lead_to_text(row, check):
    """Turn one CRM row into natural sentences (embedding models read sentences better than tables)."""
    package = row["Interested Service"]
    text = (
        f"{row['Company']} ({row['Lead ID']}) is a {row['Industry']} company in {row['City']} "
        f"with {row['Employees']} employees. "
        f"They are interested in: {package}. "
        f"Budget: {format_budget(row['Budget (USD)'], row['Budget Type'])}. "
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
        crm_check += f"; AT RISK: no contact for {check['days_since_contact']} days, contact them immediately"
    if check["special_discount"]:
        crm_check += f"; {row['Industry']} organizations get {check['special_discount']}% off"

    crm_check += f". {package} includes ONLY: {PACKAGE_FEATURES.get(package, 'unknown')}"
    smaller = cheaper_packages(package)
    if smaller:
        budget = int(row["Budget (USD)"])
        options = "; ".join(
            f"{p} ({format_price(p)}, {'within' if PACKAGE_PRICES[p] <= budget else 'above'} their budget) "
            f"includes ONLY: {PACKAGE_FEATURES[p]}"
            for p in smaller
        )
        crm_check += f". Cheaper packages of the same service: {options}"
        if not check["budget_ok"] and all(PACKAGE_PRICES[p] > budget for p in smaller):
            crm_check += f". No {package.split()[-1].lower()} package fits their budget"
    if package in WEB_AND_APP:
        crm_check += ". Cross-selling idea: Social Media Management or Ads Management after launch"
    return text + crm_check + "."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NO_INFO = "I don't have this information."

SYSTEM_PROMPT = f"""You are a Sales Assistant inside the CRM of Nova Digital Solutions,
a digital agency that offers websites, mobile apps, digital marketing and AI solutions.
You help the sales team understand their leads before contacting them.

Rules:
1. Answer using ONLY the Reference data and the conversation history.
2. Never use your own general knowledge, even if you know the answer.
3. Use the conversation history to understand follow-up questions (words like it, they, them).
4. If the user asks about the conversation itself (previous questions, their name, which leads we discussed), answer from the conversation history.
5. If the user greets you or introduces themselves, reply politely.
6. Always answer in clear natural sentences. Never copy the raw data rows.
7. When comparing a budget with a price, compare the numbers carefully. Only suggest packages that cost less than or equal to the budget.
8. If the answer is not in the Reference data or the history, say exactly: "{NO_INFO}"
9. Only mention prices, numbers, percentages and packages that appear in the Reference data. Never invent them.
10. Keep answers short (2 to 4 sentences). Do not write emails or long lists unless the user asks for them.
11. Every lead ends with a "CRM check" calculated from the playbook rules. Trust it for budget, decision maker, priority and next action.
12. You are the assistant. The user is a sales rep. Never say that the user's name is your name.
13. When you suggest a package, describe it ONLY with the features listed for it. Never give a smaller package the features of a bigger one, and say what the lead would give up.
14. For objections, follow the "Handling Objections" section of the playbook. For "the price is too high": never start with "Yes" and never agree that the price is too high; if the CRM check says the budget is enough, say their budget covers it and explain the value. Suggest a cheaper package with its price, mention the 15% bundle discount if they need more than one service, and say that project payments are split 50% when signing, 30% at the middle of the project and 20% at delivery.
15. For questions about many leads (counts, lists), use ONLY the CRM overview. Start with the exact number, never count leads yourself from single lead rows, and never invent a reason why a lead is in a list."""

SUMMARY_PROMPT = """You summarize a chat between a SALES REP (the user) and an AI ASSISTANT.
Write short bullet points. Always keep:
- the sales rep's name, written as "The sales rep's name is ...",
- every lead (company) discussed and the key facts about it,
- the questions the sales rep asked.
Do not invent anything."""

RETRY_PROMPT = """Your answer mentioned {numbers}, which do not appear in the Reference data.
Answer the same question again. Use only prices, numbers and percentages that appear in the Reference data.
If the Reference data does not contain the answer, say: "{no_info}" """

# Questions about the whole pipeline rather than one lead
_AGGREGATE = re.compile(
    r"\b(how many (?:[\w-]+ ){0,2}(?:leads|deals|companies|clients|customers)"
    r"|number of (?:[\w-]+ ){0,2}leads"
    r"|which (?:[\w-]+ ){0,2}(?:leads?|compan(?:y|ies)|clients?|customers?|deals?)"
    r"|what leads|(?:list|show) (?:me )?(?:all |the |our )?(?:[\w-]+ )?leads|all (?:the |our )?leads|our leads"
    r"|(?:hot|warm|cold|at[- ]risk|won|lost|top|best|new) (?:leads|deals)"
    r"|pipeline|who should i (?:call|contact|focus on)|call first|focus on today"
    r"|most (?:important|urgent)|prioriti[sz]\w*|priorities|overview)\b",
    re.I,
)
# Words that point back at the lead already open ("compare it with Delta Foods")
_REFERS_BACK = re.compile(r"\b(it|its|them|they|their|this lead|this one|this company|compare|vs|versus)\b", re.I)


_YES_NO_START = ("is", "are", "can", "could", "does", "do", "did", "will", "would", "should", "has", "have", "was", "were")


def clean_answer(question, answer):
    """Small fixes that do not need the model."""
    # Never open by agreeing that the price is too high ("Yes, the price is too high.")
    trimmed = re.sub(r"^(?:yes[,.!]?\s+)?(?:the\s+)?price is (?:indeed\s+)?too high[.!]\s*", "", answer, flags=re.I)
    answer = trimmed or answer
    # A question that is not yes/no should not get an answer that starts with "Yes," or "No,"
    words = question.strip().split()
    if words and words[0].lower() not in _YES_NO_START:
        answer = re.sub(r"^(?:yes|no)[,.!]?\s+", "", answer, flags=re.I) or answer
    return answer[:1].upper() + answer[1:]


def _has_numbers(text):
    return bool(money_in(text) or percents_in(text))


def _normalize(text):
    """Lower case, no accents, only letters and digits: "Café Masr!" -> " cafe masr " """
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return " " + " ".join(re.findall(r"[a-z0-9]+", text)) + " "


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

    MAX_LEADS_PER_QUESTION = 3

    def __init__(self, leads_df, pdf_chunks, embeddings, generate_fn,
                 today=None, summary_batch=3, keep_recent=2):
        self.df = leads_df.reset_index(drop=True)
        self.generate = generate_fn
        self.summary_batch = summary_batch
        self.keep_recent = keep_recent
        self.today = today or date.today()

        self.checks = [check_lead(row, self.today) for _, row in self.df.iterrows()]
        self.lead_docs = [
            Document(page_content=lead_to_text(row, check),
                     metadata={"source": "lead", "lead_id": row["Lead ID"], "company": row["Company"]})
            for (_, row), check in zip(self.df.iterrows(), self.checks, strict=False)
        ]
        self.pdf_chunks = pdf_chunks
        self.overview_doc = Document(page_content=self._overview_text(), metadata={"source": "overview"})
        self.guard = FactGuard("\n".join(chunk.page_content for chunk in pdf_chunks), PACKAGE_PRICES,
                               MONTHLY_PACKAGES, monthly_fees={"AI Chatbot": (300,)})

        # Two indexes: everything, and the playbook only.
        # When we already know the lead, we search the playbook only,
        # so rows of other companies do not confuse the model.
        self.vector_store = VectorIndex(self.pdf_chunks + self.lead_docs, embeddings)
        self.pdf_store = VectorIndex(self.pdf_chunks, embeddings)

        # Longest names first, so a long name is not matched by a shorter one inside it.
        self._names = [_normalize(name) for name in self.df["Company"]]
        self._name_order = sorted(range(len(self._names)), key=lambda i: -len(self._names[i]))
        self._ids = {str(lead_id).lower(): i for i, lead_id in enumerate(self.df["Lead ID"])}

    # ---- CRM overview (answers "how many hot leads?" without guessing) -----

    def _overview_text(self):
        priorities = Counter(check["priority"] for check in self.checks)
        stages = Counter(self.df["Stage"])
        at_risk = [i for i, check in enumerate(self.checks) if check["at_risk"]]
        lines = [
            f"CRM overview: {len(self.df)} leads in total, data as of {self.today.isoformat()}.",
            "Leads by priority: " + ", ".join(f"{p} {priorities.get(p, 0)}" for p in PRIORITIES) + ".",
            "Leads by stage: " + ", ".join(f"{s} {n}" for s, n in stages.most_common()) + ".",
            f"At-risk leads (active, no contact for more than {AT_RISK_DAYS} days): {len(at_risk)}.",
        ]

        def names(indices, limit=12):
            shown = [f"{self.df.iloc[i]['Company']} ({self.df.iloc[i]['Interested Service']})" for i in indices[:limit]]
            more = f", and {len(indices) - limit} more (see the lead list)" if len(indices) > limit else ""
            return ", ".join(shown) + more

        hot = [i for i, c in enumerate(self.checks) if c["priority"] == "Hot"]
        hot.sort(key=lambda i: -int(self.df.iloc[i]["Budget (USD)"]))
        if hot:
            lines.append(f"Hot leads, biggest budget first: {names(hot)}.")
        if at_risk:
            at_risk.sort(key=lambda i: -self.checks[i]["days_since_contact"])
            lines.append(f"At-risk leads, longest without contact first: {names(at_risk)}.")
        return "\n".join(lines)

    # ---- finding the lead -------------------------------------------------

    def find_leads(self, text):
        """
        Indexes of the leads mentioned in the text, in the order they appear.
        Matches a Lead ID (L001), the exact company name (case and accents do not matter),
        or, if nothing else matched, a company name with a small typo.
        """
        if not text:
            return []
        normalized = _normalize(text)
        found = {}  # index -> position in the text

        for match in re.finditer(r"\bl\d{3,}\b", normalized):
            index = self._ids.get(match.group())
            if index is not None:
                found.setdefault(index, match.start())

        remaining = normalized
        for i in self._name_order:
            position = remaining.find(self._names[i])
            if position >= 0:
                found.setdefault(i, position)
                # Blank out the match so a shorter name inside it is not matched again
                remaining = remaining[:position] + " " * len(self._names[i]) + remaining[position + len(self._names[i]):]

        if not found:
            fuzzy = self._fuzzy_find(normalized)
            if fuzzy is not None:
                found[fuzzy] = 0

        return sorted(found, key=found.get)

    def _fuzzy_find(self, normalized):
        """One company name written with a small typo ("nile pharam"), only if there is a single clear match."""
        words = normalized.split()
        best, best_score, tie = None, 0.0, False
        for i, name in enumerate(self._names):
            name = name.strip()
            size = len(name.split())
            if len(name) < 8:  # short names are too easy to match by accident
                continue
            for start in range(0, max(0, len(words) - size) + 1):
                window = " ".join(words[start:start + size])
                score = difflib.SequenceMatcher(None, name, window).ratio()
                if score > best_score + 1e-9:
                    best, best_score, tie = i, score, False
                elif abs(score - best_score) <= 1e-9 and i != best:
                    tie = True
        return best if best_score >= 0.9 and not tie else None

    def find_lead(self, text):
        """Index of the first lead mentioned in the text, or None."""
        leads = self.find_leads(text)
        return leads[0] if leads else None

    def resolve_leads(self, question, history, summary="", lead_hint=None):
        """
        Which leads is the sales rep talking about? In this order:
        the leads named in the question, the lead open in the CRM screen,
        the leads named in the latest message that named one, the last lead in the summary.
        """
        named = self.find_leads(question)
        if named:
            # "Compare it with Delta Foods": keep the lead that is open as well
            if lead_hint is not None and lead_hint not in named and _REFERS_BACK.search(question):
                named = [lead_hint] + named
            return named[:self.MAX_LEADS_PER_QUESTION]
        if self.is_pipeline_question(question):
            return []  # "Which lead should I call first?" is about all leads, not the open one
        if lead_hint is not None:
            return [lead_hint]
        for turn in reversed(history):
            named = self.find_leads(turn["user"])
            if named:
                return named[:self.MAX_LEADS_PER_QUESTION]
        named = self.find_leads(summary)
        return named[-1:]

    @staticmethod
    def is_pipeline_question(question):
        return bool(_AGGREGATE.search(question))

    def find_lead_in_chat(self, question, history, summary="", lead_hint=None):
        leads = self.resolve_leads(question, history, summary, lead_hint)
        return leads[0] if leads else None

    def index_of(self, lead_id):
        """Row index of a Lead ID such as "L001" (any case), or None."""
        return self._ids.get(str(lead_id).strip().lower())

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
        leads = self.resolve_leads(question, history, summary, lead_hint)

        # Add the previous question so follow-ups like "how many employees do they have?" still find something.
        search_query = (history[-1]["user"] + " " + question) if history else question

        if leads:
            docs = [self.lead_docs[i] for i in leads]
            docs += [d for d, _ in self.pdf_store.search(self.lead_docs[leads[0]].page_content, k=3)]
            docs += [d for d, _ in self.pdf_store.search(search_query, k=4)]
            if self.is_pipeline_question(question):
                docs.insert(0, self.overview_doc)
        else:
            docs = [self.overview_doc]
            docs += [d for d, _ in self.vector_store.search(question, k=4)]
            docs += [d for d, _ in self.vector_store.search(search_query, k=4)]

        # Remove duplicate chunks (the two searches often return the same ones)
        unique, seen = [], set()
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return unique, leads

    @staticmethod
    def sources_of(docs):
        """Short labels for the UI: where the answer comes from."""
        sources, seen = [], set()
        for doc in docs:
            meta = doc.metadata
            if meta.get("source") == "lead":
                key, label = ("lead", meta["lead_id"]), f"CRM: {meta['company']} ({meta['lead_id']})"
            elif meta.get("source") == "overview":
                key, label = ("overview",), "CRM overview"
            else:
                for section in meta.get("sections") or [meta.get("section", "")]:
                    key = ("playbook", section)
                    if key not in seen:
                        seen.add(key)
                        sources.append({"kind": "playbook", "label": f"Playbook p.{meta.get('page', '?')}: {section}"})
                continue
            if key not in seen:
                seen.add(key)
                sources.append({"kind": key[0], "label": label})
        return sources

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
            {"role": "user", "content": f"Old summary:\n{summary or '(none)'}\n\nNew conversation:\n{old_text}"},
        ]
        new_summary = self.generate(messages, max_new_tokens=250).strip()
        return history[self.summary_batch:], new_summary or summary

    # ---- one chat turn ----------------------------------------------------

    def answer(self, question, history, summary="", lead_hint=None):
        """
        One chat turn with all the details.
        Returns a dict: answer, history, summary, lead_index, sources, warnings.
        """
        docs, leads = self.retrieve(question, history, summary, lead_hint)
        context = "\n\n".join(doc.page_content for doc in docs)
        if leads:
            names = ", ".join(self.df.iloc[i]["Company"] for i in leads)
            context = f"Lead(s) we are talking about: {names}\n\n" + context

        messages = self.build_messages(question, history, summary, context)
        answer = self.generate(messages, max_new_tokens=300).strip() or NO_INFO

        # Fact guard: every price and percentage must come from the data.
        user_text = "\n".join([question] + [turn["user"] for turn in history])
        problems = self.guard.unsupported(answer, context, user_text)
        if problems:
            retry = messages + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": RETRY_PROMPT.format(numbers=", ".join(problems), no_info=NO_INFO)},
            ]
            second = self.generate(retry, max_new_tokens=300).strip()
            second_problems = self.guard.unsupported(second, context, user_text)
            # Use the second answer only if it is better and still answers with numbers.
            # A retry that just says "I don't have this information" would hide a mostly right answer.
            if second and len(second_problems) < len(problems) and _has_numbers(second):
                answer, problems = second, second_problems

        answer = clean_answer(question, answer)
        warnings = [f"{number} is not in the CRM or the playbook. Check it before you quote it." for number in problems]

        history = history + [{"user": question, "assistant": answer}]
        try:
            history, summary = self.summarize(history, summary)
        except Exception:  # the answer is ready; summarize again on the next turn
            logger.warning("Could not summarize the conversation this turn", exc_info=True)
        return {
            "answer": answer,
            "history": history,
            "summary": summary,
            "lead_index": leads[0] if leads else None,
            "lead_indexes": leads,
            "sources": self.sources_of(docs),
            "warnings": warnings,
        }

    def chat(self, question, history, summary="", lead_hint=None):
        """
        Simple version of answer(), used by the notebook and tests.
        Returns (answer, new_history, new_summary, lead_index).
        """
        result = self.answer(question, history, summary, lead_hint)
        return result["answer"], result["history"], result["summary"], result["lead_index"]
