import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from models.ollama import model
# from models.groq import model_groq
from retriever.chroma_ import search_documents

# FastAPI ilovasini yaratish
app = FastAPI()

# Static fayllarni ulash
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Eng mos javoblarni topish
        if request.query.strip() == "":
            # results = []
            relevant_docs = []
            return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}
        else:
            # results = collection.query(query_texts=[query], n_results=5)
            relevant_docs = search_documents(request.query, 20)

            if len(relevant_docs) == 0:
                return {"response": "Ma'lumot bazamdan ushbu savol bo'yicha  ma'lumot topilmadi. Istasangiz o'z bilimlarimdan foydalanib javob beraman"}

        context = "\n".join(relevant_docs)

        # print(context, "<javobga savol>", request.query)
        print("javobga savol ->", context)

        # Modelga savol yuborish
        prompt = f"Quyidagi kontekstdan foydalanib savolga javob bering:\n\nKontekst:\n{context}\n\nSavol: {request.query}"
        response = await model.chat(prompt)
        
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
