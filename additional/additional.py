from redis_obj.redis import redis_session
from models.groq import GroqModel
from models.ollama import OllamaModel
from retriever.chroma_ import search_documents

# sesiya orqali savolni qayta olish
async def old_context(model: GroqModel | OllamaModel, user_id: str, request: str):

    previous_session = redis_session.get_user_session(user_id) or []

    previous_context = "\n".join(previous_session)

    context_query = await model.rewrite_query(request, previous_context)

    return context_query.get('content', "")

# Eng mos javoblarni topish db dan olish
def get_docs_from_db(request: str):
    if request.strip() == "":
        relevant_docs = []
    else:
        relevant_docs = search_documents(request, 20)

    return relevant_docs

# Yangi javobni sessiyaga saqlash
def change_redis(user_id: str, query: str, response: str):
    new_session = (redis_session.get_user_session(user_id) or []) + [f"User: {query}", f"Bot: {response}"]
    redis_session.set_user_session(user_id, new_session)