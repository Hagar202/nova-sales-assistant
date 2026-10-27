"""Request and response shapes. FastAPI validates them and shows them in /docs."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Give me a short brief about Nile Pharma before my call."])
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$",
                                   description="Leave empty on the first message; reuse the returned id.")
    lead_id: str | None = Field(None, description="Lead open in the CRM screen, e.g. L001.", examples=["L001"])


class MemoryInfo(BaseModel):
    turns_in_full: int
    summary: str


class Source(BaseModel):
    kind: str = Field(..., description="lead, playbook or overview")
    label: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    lead_id: str | None = Field(None, description="The lead the answer is about.")
    lead_ids: list[str] = Field(default_factory=list, description="All leads the answer is about.")
    sources: list[Source] = Field(default_factory=list, description="Where the answer comes from.")
    warnings: list[str] = Field(default_factory=list,
                                description="Numbers in the answer that are not in the data.")
    memory: MemoryInfo
    latency_ms: int


class LeadCheck(BaseModel):
    price: str
    budget_ok: bool
    decision_maker: bool
    qualified: bool
    priority: str
    next_action: str
    days_since_contact: int
    at_risk: bool
    special_discount: int | None
    reasons: list[str]


class LeadSummary(BaseModel):
    lead_id: str
    company: str
    industry: str
    service: str
    stage: str
    priority: str
    at_risk: bool


class LeadDetails(LeadSummary):
    city: str
    employees: int
    budget: int
    budget_type: str
    contact_person: str
    job_title: str
    lead_source: str
    last_contact: str
    notes: str
    check: LeadCheck


class Health(BaseModel):
    status: str
    backend: str
    model: str
    mock: bool
    leads: int
    chunks: int
    as_of: str = Field(..., description="Date the CRM rules count from (days since contact, at risk).")
