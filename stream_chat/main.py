from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ollama import chat

app = FastAPI()

# CORS middleware ni qo‘shish
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hamma domenlarga ruxsat berish
    allow_credentials=True,
    allow_methods=["*"],  # Barcha metodlarga ruxsat berish
    allow_headers=["*"],  # Barcha headerlarga ruxsat berish
)

class ChatRequest(BaseModel):
    content: str

def sse_format(data: str):
    return f"data: {data}\n\n"

def generate_response(messages):
    for part in chat('llama3.2:3b', messages=messages, stream=True):
        yield sse_format(part['message']['content'])

@app.post("/chat/stream")
async def stream_chat(req: ChatRequest):
    messages = [{'role': 'system', 'content': "You're a helpful assistant. Answer only html tsgs. (<p>, <br>, <b>, <i>, <u>, <a>, <img>, <ul>, <ol>, <li>). No markdown"},
                {'role': 'user', 'content': "Please answer the following question using valid HTML formatting. Use <p> for paragraphs, <b> for headings, and <br> for line breaks where needed. The response should be copy-paste ready for HTML rendering. Question: {}".format(req.content) }]
    return StreamingResponse(generate_response(messages), media_type='text/event-stream')


@app.get("/")
def read_root():
    return {"message": "Welcome to the chat API!"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)