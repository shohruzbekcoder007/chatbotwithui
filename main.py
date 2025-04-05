import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from models.ollama import model
# from models.groq import model_groq
# from models.huggingface import model as model_huggingface
from retriever.chroma_ import search_documents
from redis_obj.redis import redis_session

# FastAPI ilovasini yaratish
app = FastAPI()

# Static fayllarni ulash
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str
    # user_id: str

@app.get("/")
async def read_root():
    # redis_session.delete_all_sessions()
    return FileResponse('static/index.html')
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        
        # user_id = request.user_id.strip()
        user_id = "user123"
        # Oldingi sessiyalarni olish
        previous_session = redis_session.get_user_session(user_id) or []

        # Eng mos javoblarni topish
        if request.query.strip() == "":
            relevant_docs = []
            return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}
        else:
            relevant_docs = search_documents(request.query, 20)

            # if len(relevant_docs) == 0:
            #     return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}

        # Avvalgi sessiyalarni qo‘shish
        previous_context = "\n".join(previous_session)
        context_query = await model.rewrite_query(request.query, previous_context)

        if request.query.strip() == "":
            prev_relevant_docs = []
            return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}
        else:
            prev_relevant_docs = search_documents(context_query['content'], 20)
        
        print(prev_relevant_docs, "<-prev_relevant_docs")

        context = "\n".join(relevant_docs)
        prev_context = "\n".join(prev_relevant_docs)
        # Chatbot modeliga so‘rov yuborish
        prompt = f"""
        Answer the question using the following context:
        Context: {context} {prev_context}
        Question: {request.query}
        Expanded meaning of your question: {context_query['content']}

        """

        response = await model.chat(prompt)

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
