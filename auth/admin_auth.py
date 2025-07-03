from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from models.admin import Admin, AdminLogin, AdminResponse
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
import os

router = APIRouter()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        admin = await Admin.find_one({"username": username})
        if not admin:
            raise HTTPException(status_code=401, detail="Admin not found")
        return admin
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@router.post("/api/admin/login", response_model=AdminResponse)
async def admin_login(login_data: AdminLogin):
    admin = await Admin.find_one({"username": login_data.username})
    if not admin or not pwd_context.verify(login_data.password, admin.password_hash):
        return AdminResponse(success=False, error="Invalid username or password")
    
    access_token = create_access_token({"sub": admin.username})
    return AdminResponse(success=True, token=access_token)

# Admin yaratish (faqat birinchi marta ishlatiladi)
@router.post("/api/admin/create", response_model=AdminResponse)
async def create_admin(login_data: AdminLogin):
    existing_admin = await Admin.find_one({"username": login_data.username})
    print(existing_admin)
    print("\n\n========================")
    if existing_admin:
        return AdminResponse(success=False, error="Admin already exists")
    
    hashed_password = pwd_context.hash(login_data.password)
    admin = Admin(
        username=login_data.username,
        password_hash=hashed_password
    )
    await admin.insert()
    return AdminResponse(success=True)
