from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from models.user import User, UserCreate, Token
from . import utils
from fastapi.responses import JSONResponse
import uuid
from datetime import datetime
from models.user_chat_list import UserChatList
from models.chat_message import ChatMessage

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=dict)
async def register(user_data: UserCreate):
    # Check if user exists
    existing_user = await User.find_one(User.email == user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = utils.get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        name=user_data.name
    )
    await user.insert()
    
    return {"message": "User created successfully"}

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.find_one(User.email == form_data.username)
    
    if not user or not utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Token response yaratish
    response = {"access_token": access_token, "token_type": "bearer"}
    
    # Cookie-da ham saqlash
    json_response = JSONResponse(content=response)
    json_response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=60 * utils.ACCESS_TOKEN_EXPIRE_MINUTES,
        secure=False,  # Development uchun False, production uchun True
        samesite="lax"
    )
    
    return json_response

@router.get("/me")
async def read_users_me(current_user: User | None = Depends(utils.get_current_user)):
    if not current_user:
        return {
            "email": None,
            "name": None,
            "created_at": None,
            "authenticated": False
        }
    return {
        "email": current_user.email,
        "name": current_user.name,
        "created_at": current_user.created_at,
        "authenticated": True
    }

@router.post("/api/chats/create")

async def create_chat(request: Request):
    # Har qanday holatda ham user_id bor (authenticated yoki "anonymous")
    user_id = request.state.user_id
    chat_id = str(uuid.uuid4())
    
    # Chat ID qaytarish
    print(f"Chat created, user ID: {user_id}, chat ID: {chat_id}")
    
    return {"chat_id": chat_id}

async def verify_chat_ownership(chat_id: str, user_id: str):
    # Anonymous foydalanuvchilar uchun tekshirish shart emas
    if user_id == "anonymous":
        return False
    
    try:
        # UserChatList modelidan foydalanib chat egasini tekshirish
        # Foydalanuvchining chat ro'yxatida bu chat borligini tekshirish
        user_chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if user_chat:
            return True
        
    except Exception as e:
        print(f"Error verifying chat ownership: {str(e)}")
        # Xatolik yuz berganda, xavfsizlik maqsadida False qaytarish
        return False

@router.get("/api/chat-history/{chat_id}")
async def get_chat_history(request: Request, chat_id: str):
    user_id = request.state.user_id
    verify_chat = verify_chat_ownership(chat_id, user_id)
    if(not verify_chat):
        return {"success": False, "error": "Unauthorized"}
    try:
        
        # ChatMessage dan chat_id ga tegishli xabarlarni olish va vaqt bo'yicha tartiblash
        messages = await ChatMessage.find(
            ChatMessage.chat_id == chat_id,
            ChatMessage.user_id == user_id
        ).sort(
            ChatMessage.timestamp
        ).to_list()
        
        if not messages:
            return {"success": True, "messages": [], "chat_id": chat_id}
        
        # Xabarlarni JSON formatiga o'tkazish
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": str(msg.id),
                "chat_id": msg.chat_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if hasattr(msg, "timestamp") else None,
                "metadata": msg.metadata if hasattr(msg, "metadata") else {}
            })
        
        return {
            "success": True,
            "messages": formatted_messages,
            "chat_id": chat_id
        }
    
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to retrieve chat history: {str(e)}"
        }
