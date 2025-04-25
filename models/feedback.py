from datetime import datetime
from beanie import Document
from pydantic import BaseModel, Field
from typing import Optional

class Feedback(Document):
    message_text: str
    answer_text: str | None = None
    feedback_type: str  # "like" yoki "dislike"
    comment: str | None = None
    user_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "feedbacks"

class FeedbackResponse(BaseModel):
    success: bool
    id: Optional[str] = None
    error: Optional[str] = None
