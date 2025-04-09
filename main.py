import os
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
# from models.ollama import model as model_groq
from models.groq import model as model_groq
from models.user import User
from retriever.chroma_ import search_documents
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
    client = AsyncIOMotorClient(os.getenv("DATABASE_URL", "mongodb://localhost:27017"))
    await init_beanie(
        database=client.get_default_database(),
        document_models=[User]
    )

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
        # Get user_id from authenticated user or use anonymous
        user_id = current_user.email if current_user else "anonymous"
        # Oldingi sessiyalarni olish
        previous_session = redis_session.get_user_session(user_id) or []

        # Eng mos javoblarni topish
        if request.query.strip() == "":
            relevant_docs = []
            return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}
        else:
            relevant_docs = search_documents(request.query, 5)

            # if len(relevant_docs) == 0:
            #     return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}

        # Avvalgi sessiyalarni qo‘shish
        previous_context = "\n".join(previous_session)
        context_query = await model_groq.rewrite_query(request.query, previous_context)

        if request.query.strip() == "":
            prev_relevant_docs = []
            return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}
        else:
            prev_relevant_docs = search_documents(context_query['content'], 5)

        context = "\n".join(relevant_docs)
        prev_context = "\n".join(prev_relevant_docs)
        # Chatbot modeliga so‘rov yuborish
        prompt = f"""
        Answer the question using the following context:
        Context: {context} {prev_context}
        Question: {request.query} {context_query['content']}.
        Expanded meaning of your question: {context_query['content']}
        """

        # print("Prompt ->", prompt)

        response = await model_groq.chat(prompt)

        # Huggingface modelni ishlatish
        # response_huggingface = model_huggingface.generate_text(prompt, context=context)
        # print("response_huggingface ->", response_huggingface)


        # Yangi javobni sessiyaga saqlash
        new_session = previous_session + [f"User: {request.query}", f"Bot: {response}"]
        redis_session.set_user_session(user_id, new_session)
        
        return {"response": response}
        # return {"response": response_huggingface}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
