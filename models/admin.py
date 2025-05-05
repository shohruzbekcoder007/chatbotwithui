from beanie import Document
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class Admin(Document):
    username: str
    password_hash: str
    created_at: datetime = datetime.utcnow()

    class Settings:
        name = "admins"

class AdminLogin(BaseModel):
    username: str
    password: str

class AdminResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    error: Optional[str] = None
