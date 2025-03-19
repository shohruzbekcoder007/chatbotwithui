import os
from fastapi import FastAPI
from pydantic import BaseModel
from models.ollama import model
from retriever.qdrant import search_documents

# FastAPI ilovasini yaratish
app = FastAPI()

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Eng mos javoblarni topish
        relevant_docs = await search_documents(request.query, 2)
        context = "\n".join(relevant_docs)

        # Modelga savol yuborish
        messages = [
            {"role": "system", "content": "Siz foydalanuvchilarga ilmiy ma'lumotlar beruvchi chatbot hisoblanasiz."},
            {"role": "user", "content": f"Savol: {request.query}\nKontext: {context}"}
        ]

        response = await model.invoke(messages)
        return {"response": response['content']}
    except Exception as error:
        print("Chat error:", str(error))
        return {"error": str(error)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
