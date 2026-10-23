"""Request and response shapes. FastAPI validates them and shows them in /docs."""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Give me a short brief about Nile Pharma before my call."])
    session_id: Optional[str] = Field(None, description="Leave empty on the first message; reuse the returned id.")
    lead_id: Optional[str] = Field(None, description="Lead open in the CRM screen, e.g. L001.", examples=["L001"])


class MemoryInfo(BaseModel):
    turns_in_full: int
    summary: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    lead_id: Optional[str] = Field(None, description="The lead the answer is about.")
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
    special_discount: Optional[int]
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
    model: str
    mock: bool
    leads: int
    chunks: int
