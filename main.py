import os
import time
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
# from models.langchain_groqCustom import model as model_llm
from models_llm.langchain_ollamaCustom import model as model_llm
from models.user import User
from models.feedback import Feedback
from models.admin import Admin
from models.chat_message import ChatMessage
from models.user_chat_list import UserChatList
from auth.router import router as auth_router
from auth.admin_auth import router as admin_router
from routers.chat_crud import router as chat_crud_router
from routers.feedback import router as feedback_router
from routers.admin import router as admin_page_router
from routers.chat import router as chat_router
from additional.additional import change_translate, combine_text_arrays, get_docs_from_db, change_redis, is_russian, old_context, clean_html_tags
from retriever.langchain_chroma import questions_manager
from auth.utils import SECRET_KEY, ALGORITHM, jwt
from typing import Optional
from datetime import datetime
import uuid
import psutil
import GPUtil
import asyncio
import json

# FastAPI ilovasini yaratish
app = FastAPI()

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None

# CORS sozlamalari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# User ID ni request.state ichida saqlash uchun middleware
@app.middleware("http")
async def add_user_id_to_request(request: Request, call_next):
    # Token olish va tekshirish
    user_id = "anonymous"
    token = None
    
    #     - Authorization headerdan token olish
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    #     - Cookie'dan token olish (agar header'da bo'lmasa)
    if token is None:
        token = request.cookies.get("access_token")
    
    # Token bilan autentifikatsiyani sinash
    if token:
        try:
            # JWT ni tekshirish va decode qilish
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if email:
                # User emailidan user_id ga o'tish
                user = await User.find_one(User.email == email)
                if user:
                    user_id = str(user.id)  # User ID ni olish
                else:
                    user_id = email  # User topilmasa email ishlatiladi
            else:
                print("No email found in token payload")
        except Exception as e:
            print(f"Token verification error: {e}")
    else:
        print("No token found in request")
    
    # Request state ichida user_id ni saqlash
    request.state.user_id = user_id
    
    # So'rovni davom ettirish
    response = await call_next(request)
    return response

# MongoDB va Beanie ni ishga tushirish
@app.on_event("startup")
async def startup_event():
    try:
        mongodb_url = os.getenv("DATABASE_URL")
        if not mongodb_url:
            raise ValueError("DATABASE_URL environment variable is not set")
            
        print(f"Connecting to MongoDB at: {mongodb_url}")
        
        client = AsyncIOMotorClient(
            mongodb_url,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        await client.admin.command('ping')
        print("Successfully connected to MongoDB")
        
        await init_beanie(
            database=client.get_default_database(),
            document_models=[User, Feedback, Admin, ChatMessage, UserChatList]
        )
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        raise

# Static fayllarni ulash
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.websocket("/ws/system")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # CPU, RAM
            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_total = round(ram.total / (1024 ** 3), 2)
            ram_used = round(ram.used / (1024 ** 3), 2)

            # GPU
            gpus = GPUtil.getGPUs()
            gpu_data = []
            for gpu in gpus:
                gpu_data.append({
                    "name": gpu.name,
                    "load": gpu.load * 100,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature,
                })

            data = {
                "cpu": cpu_percent,
                "ram": {
                    "percent": ram_percent,
                    "used": ram_used,
                    "total": ram_total
                },
                "gpu": gpu_data
            }

            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket disconnected: {e}")

@app.get("/")
async def read_root(request: Request, response: Response, chat_id: Optional[str] = None):
    token_value = request.cookies.get("access_token") or ""
    user_id = request.state.user_id or "anonymous"
    
    # Foydalanuvchi ID olish
    if token_value:
        try:
            payload = jwt.decode(token_value, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if email:
                # User emailidan user_id ga o'tish
                user = await User.find_one(User.email == email)
                if user:
                    user_id = str(user.id)  # User ID ni olish
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            # Token xatosi bo'lsa cookie ni tozalash
            response = RedirectResponse(url="/")
            response.delete_cookie("access_token")
            return response
    
    # Chat ID ni tekshirish
    if not chat_id:
        # Yangi chat yaratish
        new_chat_id = str(uuid.uuid4())

        # Agar foydalanuvchi login qilgan bo'lsa, chat ro'yxatiga qo'shish
        if user_id != "anonymous":
            try:
                chat = UserChatList(
                    user_id=user_id,
                    chat_id=new_chat_id,
                    name="Yangi suhbat",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await chat.insert()
            except Exception as e:
                print(f"Error creating chat in read_root: {str(e)}")
        
        response = RedirectResponse(url=f"/?chat_id={new_chat_id}")
        return response
    else:
        # Chat kimga tegishli ekanligini tekshirish
        existing_chat_owner = await UserChatList.find_one(UserChatList.chat_id == chat_id)
        
        # Anonim foydalanuvchilar faqat anonim chatlarga kirishi mumkin
        if user_id == "anonymous":
            if existing_chat_owner:
                # Agar chat boshqa foydalanuvchiga tegishli bo'lsa, yangi chat yaratish
                new_chat_id = str(uuid.uuid4())
                response = RedirectResponse(url=f"/?chat_id={new_chat_id}")
                return response
            else:
                # Agar chat hech kimga tegishli bo'lmasa, kirish mumkin
                response = FileResponse("static/index.html")
                return response

        # Autentifikatsiya qilingan foydalanuvchilar uchun chat egaligini tekshirish
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if not chat:
            # Agar chat topilmasa, yangi chat yaratish
            new_chat_id = str(uuid.uuid4())
            
            # Yangi chatni bazaga saqlash
            try:
                new_chat = UserChatList(
                    user_id=user_id,
                    chat_id=new_chat_id,
                    name="Yangi suhbat",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await new_chat.insert()
            except Exception as e:
                print(f"Error creating new chat: {str(e)}")
                
            # Yangi yaratilgan chatga yo'naltirish
            response = RedirectResponse(url=f"/?chat_id={new_chat_id}")
            return response
            
        # Chat egasi bo'lsa, sahifani ko'rsatish
        response = FileResponse("static/index.html")
        return response

@app.get("/login")
async def login_page():
    response = FileResponse('static/login.html')
    return response

# Auth routerlarini qo'shish
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(chat_crud_router)
app.include_router(feedback_router)
app.include_router(admin_page_router)
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))