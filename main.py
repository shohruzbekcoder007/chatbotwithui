import os
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from models.groq import model as model_groq
from models.user import User
from models.feedback import Feedback, FeedbackResponse
from models.admin import Admin
from redis_obj.redis import redis_session
from auth.router import router as auth_router
from auth.admin_auth import router as admin_router, get_current_admin
from auth.utils import get_current_user
from additional.additional import change_redis, get_docs_from_db, old_context

# FastAPI ilovasini yaratish
app = FastAPI()

# CORS sozlamalari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

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
            document_models=[User, Feedback, Admin]
        )
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        raise

# Static fayllarni ulash
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/yozilganlar", StaticFiles(directory="yozilganlar"), name="yozilganlar")

# Admin endpoints
@app.get("/admin")
async def admin_page():
    return FileResponse('static/admin.html')

@app.get("/api/admin/feedbacks")
async def get_feedbacks(admin: Admin = Depends(get_current_admin)):
    try:
        feedbacks = await Feedback.find_all().to_list()
        return {"success": True, "feedbacks": feedbacks}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Auth routerlarini qo'shish
app.include_router(auth_router)
app.include_router(admin_router)

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

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str
    # user_id: str

@app.get("/")
async def read_root():
    redis_session.delete_all_sessions()
    return FileResponse('static/index.html')

@app.get("/login")
async def login_page():
    return FileResponse('static/login.html')

@app.post("/chat")
async def chat(request: ChatRequest, current_user: User | None = Depends(get_current_user)):
    try:

        # user_id ni olish
        user_id = current_user.email if current_user else "anonymous"
        
        # Contextdan savolni qayta olish
        context_query = await old_context(model_groq, user_id, request.query)

        # Eng mos javoblarni topish old_context
        prev_relevant_docs = get_docs_from_db(context_query)
        
        # Eng mos javoblarni topish
        relevant_docs = get_docs_from_db(request.query)

        context = "\n".join(relevant_docs)
        prev_context = "\n".join(prev_relevant_docs)

        prompt = f"""
        Answer the question using the following context:
        Context: {context}
        Question: {request.query}.
        """

        prompt_old = f"""
        Answer the question using the following context:
        Context: {prev_context}
        Question: {context_query}.
        """

        response_current = await model_groq.chat(prompt)
        response_old = await model_groq.chat(prompt_old)

        # 3 ta savolni yaratish
        questions = await model_groq.generate_questions(request.query)
        # print(questions, "<--")

        response_add = await model_groq.logical_context(f"{response_current}\n ||| \n{response_old}")
        
        # Yangi javobni sessiyaga saqlash
        change_redis(user_id, request.query, response_add['content'])
        
        return {"response": response_add['content']}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
