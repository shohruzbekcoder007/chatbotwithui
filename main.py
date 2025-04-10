import os
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
# from models.ollama import model as model_groq
from additional.additional import change_redis, get_docs_from_db, old_context
from models.groq import model as model_groq
from models.user import User
from redis_obj.redis import redis_session
from auth.router import router as auth_router
from auth.utils import get_current_user

# FastAPI ilovasini yaratish
app = FastAPI()

# CORS sozlamalari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
            document_models=[User]
        )
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        raise

# Static fayllarni ulash
app.mount("/static", StaticFiles(directory="static"), name="static")

# Auth routerini qo'shish
app.include_router(auth_router)

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

        # Chatbot modeliga so‘rov yuborish
        # prompt = f"""
        # Answer the question using the following context:
        # Context: {context} {prev_context}
        # Question: {request.query} {context_query}.
        # Expanded meaning of your question: {context_query}
        # """

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

        # print("Prompt ->", prompt)

        response_current = await model_groq.chat(prompt)
        response_old = await model_groq.chat(prompt_old)
        
        response_add = await model_groq.logical_context(f"{response_current}\n ||| \n{response_old}")
        
        # response = f"{response_current}\n ||| \n{response_old} \n ||| \n{response_add['content']}"
        
        # Yangi javobni sessiyaga saqlash
        change_redis(user_id, request.query, response_add['content'])
        
        return {"response": response_add['content']}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
