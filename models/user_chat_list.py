from datetime import datetime
from beanie import Document
from pydantic import Field
from typing import Optional

class UserChatList(Document):
    """Foydalanuvchi chat ro'yxatini saqlash uchun model"""
    user_id: str  # Foydalanuvchi ID
    chat_id: str  # Chat ID (UUID)
    name: str = "Yangi suhbat"  # Chat nomi
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "user_chat_list"  # MongoDB collection nomi 