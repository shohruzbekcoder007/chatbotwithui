from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    password: str
    name: str | None = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None
