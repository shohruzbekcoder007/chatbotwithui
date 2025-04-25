from datetime import datetime
from beanie import Document
from pydantic import Field
from typing import Optional

class ChatMessage(Document):
    """Chat xabarlarini saqlash uchun model"""
    user_id: str  # Foydalanuvchi ID (anonymous yoki haqiqiy)
    chat_id: str  # Chat ID (UUID)
    message: str  # Foydalanuvchi savoli
    response: str  # AI javobi
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "chat_messages"  # MongoDB collection nomi 