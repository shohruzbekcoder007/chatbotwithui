from datetime import datetime
from typing import Optional
from beanie import Document, Indexed
from pydantic import BaseModel, EmailStr

class User(Document):
    email: Indexed(str, unique=True)  # Indexed field for email
    hashed_password: str
    name: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()

    class Settings:
        name = "users"  # Collection name in MongoDB

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "name": "John Doe"
            }
        }

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
