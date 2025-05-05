from retriever.langchain_chroma import search_documents
from redis_obj.redis import redis_session
from models.langchain_ollama import rewrite_query
from typing import List
import unicodedata
from translate.translate import translator


# Eng mos javoblarni topish db dan olish
def get_docs_from_db(request: str):
    try:
        if not request or request.strip() == "":
            return []
        
        relevant_docs = search_documents(request, 4)
        if not relevant_docs:
            return []
            
        return relevant_docs
    except Exception as e:
        print(f"Error in get_docs_from_db: {str(e)}")
        return []

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

def is_russian(text: str) -> bool:
    """
    Check if the text contains Cyrillic characters
    """
    for c in text:
        try:
            if 'CYRILLIC' in unicodedata.name(c):
                return True
        except ValueError:
            pass
    return False

async def change_translate(text: str, lang: str) -> str:
    try:
        if not text:
            return ""

        natija = await translator.russian_to_uzbek("Статья 6") or ""
        print(natija, "<- natija") 
        if lang == "uz":
            return await translator.russian_to_uzbek(text) or ""
        elif lang == "ru":
            return await translator.uzbek_to_russian(text) or ""
        else:
            return text
    except Exception as e:
        print(f"Tarjima xatosi: {str(e)}")
        return text  # Original matnni qaytarish
