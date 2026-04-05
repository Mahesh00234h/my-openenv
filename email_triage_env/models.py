from pydantic import BaseModel


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
    score: float  # [0.0, 1.0]
    category_score: float
    priority_score: float
    response_score: float
    explanation: str
