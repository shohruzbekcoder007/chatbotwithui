import os
from fastapi import FastAPI, WebSocket, Request, Depends, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
# from models.langchain_groqCustom import model as model_llm
from models.langchain_ollamaCustom import model as model_llm
from models.user import User
from models.feedback import Feedback
from models.admin import Admin
from models.chat_message import ChatMessage
from models.user_chat_list import UserChatList
from auth.router import router as auth_router
from auth.admin_auth import router as admin_router, get_current_admin
from additional.additional import combine_text_arrays, get_docs_from_db, change_redis, old_context
from auth.utils import SECRET_KEY, ALGORITHM, jwt
from typing import Optional
from datetime import datetime
import uuid
import psutil
import GPUtil
import asyncio
import json

from llm_models.google_gemma27b import GemmaModel

# Model obyektini yaratish
gemma = GemmaModel()
gemma.start_processing()

# FastAPI ilovasini yaratish
app = FastAPI()

# Templates
templates = Jinja2Templates(directory="static")

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
    
    # 1. Authorization headerdan token olish
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    # 2. Cookie'dan token olish (agar header'da bo'lmasa)
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
# app.mount("/tayyor_json", StaticFiles(directory="tayyor_json"), name="tayyor_json")

# Admin endpoints
@app.get("/admin")
async def admin_page():
    return FileResponse('static/admin.html')

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

@app.get("/api/admin/feedbacks")
async def get_feedbacks(admin: Admin = Depends(get_current_admin)):
    try:
        feedbacks = await Feedback.find_all().to_list()
        return {"success": True, "feedbacks": feedbacks}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Feedback endpoint
@app.post("/api/feedback")
async def submit_feedback(request: Request):
    try:
        data = await request.json()
        feedback = Feedback(
            message_text=data.get("message_text"),
            answer_text=data.get("answer_text"),
            feedback_type=data.get("feedback_type"),
            comment=data.get("comment", ""),
            user_id=data.get("user_id")
        )
        await feedback.insert()
        return {"success": True, "id": str(feedback.id)}
    except Exception as e:
        return {"success": False, "error": str(e)}

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

@app.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        # user_id ni olish (request.state.user_id dan)
        user_id = request.state.user_id
        print(f"Using user_id from request.state: {user_id}")
        
        # Chat ID ni olish (agar berilgan bo'lsa)
        chat_id = chat_request.chat_id    
        print(f"Using chat ID: {chat_id}")
        
        # Contextdan savolni qayta olish
        context_query = await old_context(model_llm, user_id, chat_request.query)

        print(context_query, "<<-context_query")
        
        relevant_docs = get_docs_from_db(chat_request.query)
        relevant_docs_add = get_docs_from_db(context_query)

        # ChromaDB natijalaridan faqat matnlarni olish
        docs = relevant_docs.get("documents", []) if isinstance(relevant_docs, dict) else []
        docs_add = relevant_docs_add.get("documents", []) if isinstance(relevant_docs_add, dict) else []

        unique_results = await combine_text_arrays(docs[0], docs_add[0])

        print(f"Unique results count: {len(unique_results)}", unique_results)

        # context1 = "Kontekst matni"
        # query1 = "O'zbekiston poytaxti qaysi shahar?"
        # response1 = await gemma.chat(context1, query1)
        # print(f"Model javobi: {response1}")

        # return {"success": True, "response": response1}

        context = "\n".join(unique_results)

        try:
            response_current = await model_llm.chat(context, chat_request.query)
        except Exception as chat_error:
            print(f"Error in chat model: {str(chat_error)}")
            return {"error": str(chat_error)}
        
        # Yangi javobni sessiyaga saqlash
        change_redis(user_id, chat_request.query, response_current)
        
        # Savol va javobni MongoDB ga saqlash (faqat autentifikatsiya qilingan foydalanuvchilar uchun)
        if user_id != "anonymous":
            chat_message = ChatMessage(
                user_id=user_id,
                chat_id=chat_id,
                message=chat_request.query,
                response=response_current,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await chat_message.insert()
            print(f"Saved message to MongoDB with ID: {chat_message.id}")
            
            # UserChatList jadvaliga tekshirish va yozish
            existing_chat = await UserChatList.find_one(
                UserChatList.user_id == user_id,
                UserChatList.chat_id == chat_id
            )
            
            if not existing_chat:
                # Chatni foydalanuvchi ro'yxatiga qo'shish
                user_chat = UserChatList(
                    user_id=user_id,
                    chat_id=chat_id,
                    name=f"Suhbat: {chat_request.query[:30]}...",  # Birinchi savoldan nom yaratish
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await user_chat.insert()
                print(f"Added chat to user's chat list: {chat_id}")
            else:
                # Mavjud yozuvning updated_at vaqtini yangilash
                await existing_chat.set({
                    "updated_at": datetime.utcnow()
                })
                print(f"Updated timestamp for existing chat: {chat_id}")
        else:
            print(f"Anonymous user, message not saved to database. Chat ID: {chat_id}")

        return {"response": response_current}

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}

@app.get("/api/chat-history/{chat_id}")
async def get_chat_history(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Faqat o'z suhbatlarini ko'rish imkonini berish
        messages = await ChatMessage.find(
            ChatMessage.chat_id == chat_id,
            ChatMessage.user_id == user_id
        ).sort("+created_at").to_list()
        
        # Xabarlarni formatlash
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": str(msg.id),
                "message": msg.message,
                "response": msg.response,
                "created_at": msg.created_at.isoformat(),
                "updated_at": msg.updated_at.isoformat()
            })
        
        return {"success": True, "messages": formatted_messages}
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return {"success": False, "error": str(e)}

@app.put("/api/chat-message/{message_id}")
async def update_chat_message(request: Request, message_id: str, update_data: dict):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Xabarni topish
        message = await ChatMessage.get(message_id)
        
        # Xabar topilmasa yoki boshqa foydalanuvchiga tegishli bo'lsa, xatolik
        if not message or message.user_id != user_id:
            return {"success": False, "error": "Message not found or access denied"}
        
        # Xabarni yangilash
        update_fields = {}
        
        if "message" in update_data:
            update_fields["message"] = update_data["message"]
            
        if "response" in update_data:
            update_fields["response"] = update_data["response"]
        
        # updated_at ni har doim yangilash
        update_fields["updated_at"] = datetime.utcnow()
        
        # Yangilash
        await message.set({**update_fields})
        
        return {"success": True, "message": "Message updated successfully"}
    except Exception as e:
        print(f"Error updating chat message: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/user-chats")
async def get_user_chats(request: Request):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun bo'sh ro'yxat
        if user_id == "anonymous":
            return {"success": True, "chats": []}
        
        # Foydalanuvchining barcha suhbatlarini olish
        user_chats = await UserChatList.find(
            UserChatList.user_id == user_id
        ).sort("-updated_at").to_list()  # Eng so'nggi yangilangan birinchi
        
        # Formatlash
        formatted_chats = []
        for chat in user_chats:
            formatted_chats.append({
                "chat_id": chat.chat_id,
                "name": chat.name,
                "created_at": chat.created_at.isoformat(),
                "updated_at": chat.updated_at.isoformat()
            })
        
        return {"success": True, "chats": formatted_chats}
    except Exception as e:
        print(f"Error retrieving user chats: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/chat/{chat_id}")
async def get_chat(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            return {"success": False, "error": "Anonymous users cannot access chat details"}
        
        # Chat ma'lumotini olish
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if not chat:
            return {"success": False, "error": "Chat not found"}
        
        # Chat xabarlarini olish
        messages = await ChatMessage.find(
            ChatMessage.chat_id == chat_id,
            ChatMessage.user_id == user_id
        ).sort("+created_at").to_list()
        
        # Natijani formatlash
        result = {
            "chat_id": chat.chat_id,
            "name": chat.name,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat(),
            "messages": []
        }
        
        # Xabarlarni qo'shish
        for msg in messages:
            result["messages"].append({
                "id": str(msg.id),
                "message": msg.message,
                "response": msg.response,
                "created_at": msg.created_at.isoformat(),
                "updated_at": msg.updated_at.isoformat()
            })
        
        return {"success": True, "data": result}
    except Exception as e:
        print(f"Error retrieving chat: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/api/chat/{chat_id}/rename")
async def rename_chat(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            return {"success": False, "error": "Anonymous users cannot rename chats"}
        
        # JSON ma'lumotlarini olish
        data = await request.json()
        new_name = data.get("name", "Yangi suhbat")
        
        # UserChatList jadvalidagi chat nomini yangilash
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if chat:
            await chat.set({
                "name": new_name,
                "updated_at": datetime.utcnow()
            })
            return {"success": True, "message": "Chat renamed successfully"}
        else:
            return {"success": False, "error": "Chat not found"}
    except Exception as e:
        print(f"Error renaming chat: {str(e)}")
        return {"success": False, "error": str(e)}

@app.delete("/api/chat/{chat_id}")
async def delete_chat(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            return {"success": False, "error": "Anonymous users cannot delete chats"}
        
        # UserChatList jadvalidan o'chirish
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if chat:
            await chat.delete()
            
            # ChatMessage jadvalidagi xabarlarni ham o'chirish
            await ChatMessage.find(
                ChatMessage.user_id == user_id,
                ChatMessage.chat_id == chat_id
            ).delete_many()
            
            return {"success": True, "message": "Chat and all messages deleted"}
        else:
            return {"success": False, "error": "Chat not found"}
    except Exception as e:
        print(f"Error deleting chat: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/api/chat")
async def create_chat(request: Request):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            return {"success": False, "error": "Anonymous users cannot create chats"}
        
        # Yangi chat ID yaratish
        new_chat_id = str(uuid.uuid4())
        
        # UserChatList ga yangi chat qo'shish
        chat = UserChatList(
            user_id=user_id,
            chat_id=new_chat_id,
            name="Yangi suhbat",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await chat.insert()
        
        return {"success": True, "chat_id": new_chat_id}
    except Exception as e:
        print(f"Error creating chat: {str(e)}")
        return {"success": False, "error": str(e)}

# Auth routerlarini qo'shish
app.include_router(auth_router)
app.include_router(admin_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))