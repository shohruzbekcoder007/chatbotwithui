from datetime import datetime, timedelta
import os
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from models.user import User, TokenData

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-keep-it-secret")  # Change this in production
ALGORITHM = "HS256"
# Token expiration time in minutes (7 days = 10080 minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 10080

# bcrypt o'rniga pbkdf2_sha256 sxemasidan foydalanish
# Bu Python standart kutubxonasida mavjud, qo'shimcha bog'liqliklar talab qilmaydi
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    default="pbkdf2_sha256",
    pbkdf2_sha256__default_rounds=30000,  # Optimal xavfsizlik darajasi
    deprecated="auto"
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> User | None:
    if not token:
        return None
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
            
        # Get user from database
        user = await User.find_one(User.email == email)
        if user is None:
            return None
            
        return user
    except JWTError:
        return None

# Cookie orqali foydalanuvchini olish
async def get_user_from_cookie(request) -> User | None:
    token = request.cookies.get("access_token")
    
    if not token:
        return None
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
            
        # Get user from database
        user = await User.find_one(User.email == email)
        if user is None:
            return None
            
        return user
    except JWTError:
        return None
