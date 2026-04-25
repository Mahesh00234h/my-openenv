from pydantic import BaseModel, field_validator


class Observation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str  # ISO-8601
    inbox_position: int  # 0-indexed
    total_emails: int


class Action(BaseModel):
    email_id: str
    category: str  # spam | urgent | newsletter | support | internal
    priority: str  # low | medium | high | urgent
    suggested_response: str


class Reward(BaseModel):
    score: float  # strictly (0.0, 1.0) exclusive
    category_score: float
    priority_score: float
    response_score: float
    explanation: str

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Ensure score is strictly between 0 and 1 (exclusive). Force rebuild v2."""
        return max(0.001, min(0.999, float(v)))

from typing import Literal, Optional
from enum import Enum


class EpisodePhase(str, Enum):
    EMAIL = "email"
    CALENDAR = "calendar"
    DELEGATION = "delegation"
    DONE = "done"


class CalendarAction(BaseModel):
    conflict_id: str
    resolution: Literal["reschedule_a", "reschedule_b", "cancel_a", "cancel_b", "delegate_a"]
    rationale: Optional[str] = ""


class DelegationAction(BaseModel):
    task_id: str
    assignee: Literal["self", "junior", "manager", "external", "drop"]
    delegation_message: Optional[str] = ""


class ChiefOfStaffAction(BaseModel):
    """Single unified action model. Only the fields matching the current phase matter."""
    phase: EpisodePhase

    # email phase fields
    email_id: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    suggested_response: Optional[str] = ""

    # calendar phase fields
    conflict_id: Optional[str] = None
    resolution: Optional[str] = None
    rationale: Optional[str] = ""

    # delegation phase fields
    task_id: Optional[str] = None
    assignee: Optional[str] = None
    delegation_message: Optional[str] = ""


class CalendarConflict(BaseModel):
    conflict_id: str
    event_a: dict
    event_b: dict
    conflict_reason: str
    resolution_options: list
    conflict_number: int
    total_conflicts: int


class DelegationTask(BaseModel):
    task_id: str
    title: str
    source_email_id: str
    urgency: str
    requires_technical: bool
    task_number: int
    total_tasks: int
