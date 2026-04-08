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
        """Ensure score is strictly between 0 and 1 (exclusive)."""
        return max(0.001, min(0.999, v))
