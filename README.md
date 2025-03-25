# Python RAG Ilovasi

Bu loyiha Node.js RAG ilovasining Python versiyasi hisoblanadi. Ilova FastAPI frameworkida yozilgan bo'lib, Hugging Face transformers va Qdrant vektorli ma'lumotlar bazasidan foydalanadi.

## O'rnatish

1. Loyihani clone qiling:
```bash
git clone <repository_url>
cd pythonrag
```

2. Virtual muhit yarating va faollashtiring:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac uchun
venv\Scripts\activate     # Windows uchun
```

3. Kerakli paketlarni o'rnating:
```bash
pip install -r requirements.txt
```

4. `.env` faylini yarating va sozlamalarni kiriting:
```
OPENAI_API_KEY=your_api_key_here
PORT=5000
```

## Ishga tushirish

Serverni ishga tushirish uchun:

```bash
python main.py
```

Server http://localhost:5000 manzilida ishga tushadi.

## API Endpointlari

### POST /chat

Chat endpointi orqali savollar yuborish mumkin.

Request:
```json
{
    "query": "Savol matni"
}
```

Response:
```json
{
    "response": "Javob matni"
}
```

## Texnologiyalar

- FastAPI
- Hugging Face Transformers
- Python-dotenv
- Chromadb
- Langchain
- Langchain-community
- Langchain-core
- Sentence-transformers
- Numpy
