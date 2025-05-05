from retriever.langchain_chroma import search_documents
from redis_obj.redis import redis_session
from models.langchain_ollama import rewrite_query
from typing import List

# Eng mos javoblarni topish db dan olish
def get_docs_from_db(request: str):
    if request.strip() == "":
        relevant_docs = []
    else:
        relevant_docs = search_documents(request, 4)

    return relevant_docs

# Yangi javobni sessiyaga saqlash
def change_redis(user_id: str, query: str, response: str):

    # Oldingi sessiyani olish
    current_session = redis_session.get_user_session(user_id) or []
    
    # Yangi so'rov va javobni qo'shish
    new_session = current_session + [f"User: {query}, Bot: {response}"]
    
    # Faqat oxirgi 3 ta almashinuvni (6 ta element - 3 ta so'rov va 3 ta javob) saqlash
    if len(new_session) > 3:
        new_session = new_session[-3:]
    
    # Yangilangan sessiyani saqlash
    redis_session.set_user_session(user_id, new_session)

# sesiya orqali savolni qayta olish
async def old_context(user_id: str, request: str):

    previous_session = redis_session.get_user_session(user_id) or []

    if(len(previous_session) > 0):
        previous_session = filter_salutations(previous_session)

    previous_context = "\n".join(previous_session)

    context_query = await rewrite_query(request, previous_context)

    if isinstance(context_query, dict):
        return context_query.get('content', request)
    else:
        return context_query or request

def filter_salutations(results):
    return [r for r in results if not any(kw in r.lower() for kw in ['salom', 'assalomu alaykum', 'hurmatli'])]

async def combine_text_arrays(text1: List[str], text2: List[str]) -> List[str]:
    """
    2 ta text arrayni combain qiliadigan yan'ni bir xillaridan faqat 1 ta qoldiradigan function
    """
    # Set orqali duplicatelarni o'chiramiz
    unique_texts = set(text1 + text2)
    return list(unique_texts)
