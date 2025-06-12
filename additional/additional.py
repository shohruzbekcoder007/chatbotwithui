import re
from retriever.langchain_chroma import search_documents
from redis_obj.redis import redis_session
from models_llm.langchain_ollamaCustom import model as model_llm
from typing import List
from translate.translate import translator

def clean_html_tags(text: str) -> str:
    """
    Matn ichidagi HTML teglarni tozalab beradi
    
    Args:
        text (str): Tozalanishi kerak bo'lgan matn
        
    Returns:
        str: HTML teglardan tozalangan matn
    """
    if not text:
        return ""
        
    # HTML teglarni tozalash (<b>, </b>, <i>, </i>, <p>, </p> va boshqalar)
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    
    # Ko'p bo'sh joylarni bitta bo'sh joy bilan almashtirish
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Boshidagi va oxiridagi bo'sh joylarni tozalash
    return cleaned_text.strip()


# Eng mos javoblarni topish db dan olish
def get_docs_from_db(request: str):
    try:
        if not request or request.strip() == "":
            return []
        
        relevant_docs = search_documents(request, 100)

        # Write relevant_docs to a text file
        with open("relevant_docs.txt", "w", encoding="utf-8") as file:
            file.write(str(relevant_docs))
        
        if not relevant_docs:
            return []
            
        return relevant_docs
    except Exception as e:
        print(f"Error in get_docs_from_db: {str(e)}")
        return []

# Yangi javobni sessiyaga saqlash
async def change_redis(user_id: str, query: str, response: str, chat_id: str = None):
    """
    Foydalanuvchi savoli va javobini Redis-ga saqlash
    
    Args:
        user_id: Foydalanuvchi ID si
        query: Foydalanuvchi savoli
        response: AI javob matni
        chat_id: Chat ID si (ixtiyoriy)
    """
    try:
        # Agar chat_id berilmagan bo'lsa
        if not chat_id:
            chat_id = "default"
            
        # Redis set_question_session funksiyasini chaqirish - await is needed
        await redis_session.set_question_session(user_id, chat_id, query, response)
        return True
    except Exception as e:
        print(f"Redis-ga saqlashda xatolik: {str(e)}")
        return False

async def change_redis_question(user_id: str, question: str, chat_id: str):
    """
    Foydalanuvchi savolini Redis-ga saqlash
    
    Args:
        user_id: Foydalanuvchi ID si
        question: Foydalanuvchi savoli
        chat_id: Chat ID si
    """
    try:
        # Oldingi sessiyani olish
        current_session = await redis_session.get_user_session(f"{user_id}_{chat_id}") or {}
        
        # Savollar ro'yxatini olish
        questions = current_session.get("questions", [])
        
        # Yangi savolni qo'shish
        questions.append(question)
        
        # Faqat oxirgi 5 ta savolni saqlash
        if len(questions) > 5:
            questions = questions[-5:]
        
        # Yangilangan ma'lumotlarni saqlash
        current_session["questions"] = questions
        await redis_session.set_user_session(f"{user_id}_{chat_id}", current_session)
        
        return True
    except Exception as e:
        print(f"Savolni Redis-ga saqlashda xatolik: {str(e)}")
        return False

# sesiya orqali savolni qayta olish
async def old_context(user_id: str, request: str, chat_id: str = None):
    """
    Oldingi savol-javoblar kontekstini hisobga olgan holda so'rovni qayta yozish
    
    Args:
        user_id: Foydalanuvchi ID si
        request: Joriy savol
        chat_id: Chat ID si (ixtiyoriy)
        
    Returns:
        str: Kontekst bilan to'ldirilgan savol
    """
    try:
        # Redis-dan savol-javoblar tarixini olish - async method should be awaited
        chat_history = await redis_session.get_question_session(user_id, chat_id)
        
        if not chat_history:
            return request
            
        # Salomlashuvlarni filtrlash
        filtered_history = []
        for item in chat_history:
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            if not any(kw in question.lower() for kw in ['salom', 'assalomu alaykum', 'hurmatli']):
                filtered_history.append(f"User: {question}\nAI: {answer}")
                
        # Tarixni matn formatiga o'girish
        previous_context = "\n\n".join(filtered_history)
        
        # Savol qayta yozish funksiyasini chaqirish va natijani kutish
        context_query = await model_llm.rewrite_query(request, previous_context)
        
        if isinstance(context_query, dict):
            return context_query.get('content', request)
        else:
            return context_query or request
    except Exception as e:
        print(f"Savolni kontekst bilan qayta yozishda xatolik: {str(e)}")
        return request

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
    cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))
    
    if cyrillic_count > latin_count:
        return True
    elif latin_count > cyrillic_count:
        return False
    else:
        return False
    

async def change_translate(text: str, to_lang: str) -> str:
    try:
        if not text:
            print("Tarjima qilish uchun matn bo'sh")
            return ""

        if to_lang == "uz":
            print("------------------------------- Tarjima qilinmoqda uzbek tiliga")
            return translator._russian_to_uzbek_sync(text) or ""
        elif to_lang == "ru":
            print("------------------------------- Tarjima qilinmoqda rus tiliga")
            return translator._uzbek_to_russian_sync(text) or ""
        else:
            return text
    except Exception as e:
        print(f"Tarjima xatosi: {str(e)}")
        return text  # Original matnni qaytarish
