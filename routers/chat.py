from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from datetime import datetime
import time
from models.chat_message import ChatMessage
from models.user_chat_list import UserChatList
from additional.additional import change_translate, combine_text_arrays, get_docs_from_db, change_redis, is_russian, old_context, clean_html_tags
from models_llm.langchain_ollamaCustom import model as model_llm
from retriever.langchain_chroma import questions_manager
from pydantic import BaseModel
from typing import Optional

# Router yaratish
router = APIRouter(prefix="", tags=["chat"])

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None

@router.post("/chat")
async def chat(request: Request, chat_request: ChatRequest):
    try:
        # user_id ni olish (request.state.user_id dan)
        user_id = request.state.user_id
        print(f"Using user_id from request.state: {user_id}")
        
        # Chat ID ni olish (agar berilgan bo'lsa)
        chat_id = chat_request.chat_id
        # print(f"Using chat ID: {chat_id}")

        question = chat_request.query
        language = 'uz'

        # if len(chat_request.query.split(" ")) < 2:
            # res = "Iltimos savolingizni to'liqroq kiriting"
            # print(res)
            # return {"response": res}

        if(is_russian(question)):
            question = await change_translate(question, "uz")
            language = 'ru'

        # print(chat_request.query, "<<- orginal question\n", question, "<<-question\n", language, "<<-language")
        
        # Contextdan savolni qayta olish
        context_query = await old_context(user_id, question)

        # print(context_query, "<<- 2 - context_query")
        
        relevant_docs = get_docs_from_db(question)
        relevant_docs_add = get_docs_from_db(context_query)

        # ChromaDB natijalaridan faqat matnlarni olish
        docs = relevant_docs.get("documents", []) if isinstance(relevant_docs, dict) else []
        docs_add = relevant_docs_add.get("documents", []) if isinstance(relevant_docs_add, dict) else []

        # print(docs, "<<-docs")
        # print(docs_add, "<<-docs_add")

        unique_results = await combine_text_arrays(docs[0], docs_add[0])
        
        # print(f"Unique results count: {len(unique_results)}", unique_results)

        context = "\n- ".join(unique_results)
        
        try:
            response_current = await model_llm.chat(context, (question or chat_request.query) + context_query, language)
        except Exception as chat_error:
            print(f"Error in chat model: {str(chat_error)}")
            return {"error": str(chat_error)}
        
        # Yangi javobni sessiyaga saqlash
        change_redis(user_id, chat_request.query, response_current)
        
        # Savol va javobni MongoDB ga saqlash (faqat autentifikatsiya qilingan foydalanuvchilar uchun)
        if user_id != "anonymous":
            chat_message = ChatMessage(
                user_id=user_id,
                chat_id=chat_id,
                message=chat_request.query,
                response=response_current,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            await chat_message.insert()
            print(f"Saved message to MongoDB with ID: {chat_message.id}")
            
            # UserChatList jadvaliga tekshirish va yozish
            existing_chat = await UserChatList.find_one(
                UserChatList.user_id == user_id,
                UserChatList.chat_id == chat_id
            )
            
            if not existing_chat:
                # Chatni foydalanuvchi ro'yxatiga qo'shish
                user_chat = UserChatList(
                    user_id=user_id,
                    chat_id=chat_id,
                    name=f"Suhbat: {chat_request.query[:30]}...",  # Birinchi savoldan nom yaratish
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await user_chat.insert()
                print(f"Added chat to user's chat list: {chat_id}")
            else:
                # Mavjud yozuvning updated_at vaqtini yangilash
                await existing_chat.set({
                    "updated_at": datetime.utcnow()
                })
                print(f"Updated timestamp for existing chat: {chat_id}")
        else:
            print(f"Anonymous user, message not saved to     - Chat ID: {chat_id}")

        return {"response": response_current}

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}

@router.post("/chat/stream")
async def stream_chat(request: Request, req: ChatRequest):
    chat_id = req.chat_id
    print(f"---------------- REQUEST: {req} ----------------")
    if not chat_id:
        return {"error": "Chat ID is missing. Please provide a valid chat ID."}

    question = req.query
    language = 'uz'

    user_id = request.state.user_id
    # print(f"Using user_id from request.state: {user_id}")

    suggestion_text = "\n <br><br><b> Tavsiya qilingan savol: </b> " 

    if(is_russian(question)):
        question = await change_translate(question, "uz")
        language = 'ru'
        suggestion_text = "\n <br><br><b> Предложенный вопрос: </b> "


    context_query = await old_context(user_id, question)

    # Kontekstni tayyorlash
    relevant_docs = get_docs_from_db(question)
    relevant_docs_add = get_docs_from_db(context_query)

    docs = relevant_docs.get("documents", []) if isinstance(relevant_docs, dict) else []
    docs_add = relevant_docs_add.get("documents", []) if isinstance(relevant_docs_add, dict) else []

    unique_results = await combine_text_arrays(docs[0], docs_add[0])

    context = "\n- ".join(unique_results) if unique_results else ""

    print(f"Context:\n {context}")

    async def event_generator():
        response_current = ""
        
        # # MongoDB-dan oldingi xabarlarni olish
        previous_messages = await ChatMessage.find(
            ChatMessage.user_id == user_id,
            ChatMessage.chat_id == chat_id
        ).sort("+created_at").to_list()
        
        # # Oldingi mavjud savollarni yig'ish (tavsiya qilingan savollarni filtrlash uchun)
        previous_questions = set()
        for msg in previous_messages:
            # Asosiy savollarni qo'shish
            if msg.message:
                previous_questions.add(msg.message.strip().lower())
    
        # print(f"Oldingi savollar: {previous_questions}")
    
        # Modeldan chat javobi olish
        async for token in model_llm.chat_stream(context=context, query=question, language=language):
            response_current += token
            yield f"{token}\n\n"
        
        # yield suggestion_text
        
        questions_manager_questions = questions_manager.search_documents(response_current, 10)
        suggested_question = ""
        if questions_manager_questions:
            sq_docs = questions_manager_questions.get("documents", []) if isinstance(questions_manager_questions, dict) else []
            print(sq_docs, "<<- sq_docs")
            question_docs = []
            previous_questions.add(req.query.strip().lower())
            print(f"\n\nOldingi savollar: {list(previous_questions)}")
            for doc in sq_docs[0]:
                # Tavsiya qilingan savollarni oldingi savollar bilan solishtirish
                # print(f"\n - Previous question: {doc.lower()} - {doc.lower() in list(previous_questions)}")
                if doc.lower() not in list(previous_questions):
                    question_docs.append(doc)
                    
            suggestion_context = "\n- ".join(question_docs) if question_docs else ""
            print(f"\n\nSuggestion question context: {suggestion_context}")

            # Stream orqali tavsiya qilingan savollarni olish
            async for token in model_llm.get_stream_suggestion_question(suggestion_context, question, response_current, language):
                suggested_question += token
                # yield f"{token}\n\n"  # Un-commenting this line to yield tokens

            suggestion_result = suggestion_text + suggested_question

            for token in suggestion_result:
                time.sleep(0.006)  # Tokenlarni chiqarish uchun kutish
                yield f"{token}\n\n"
        
        # MongoDB ga saqlash
        chat_message = ChatMessage(
            user_id=user_id,
            chat_id=chat_id,
            message=req.query,
            response=response_current,
            suggestion_question=clean_html_tags(suggested_question),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await chat_message.insert()
        print(f"Saved message to MongoDB with ID: {chat_message.id}")

        # Faqat login qilgan foydalanuvchilar uchun chat ro'yxatiga qo'shish
        if user_id != "anonymous":
            existing_chat = await UserChatList.find_one(
                UserChatList.user_id == user_id,
                UserChatList.chat_id == chat_id
            )

            if not existing_chat:
                user_chat = UserChatList(
                    user_id=user_id,
                    chat_id=chat_id,
                    name=f"Suhbat: {req.query[:30]}...",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await user_chat.insert()
                print(f"Added chat to user's chat list: {chat_id}")
            else:
                await existing_chat.set({"updated_at": datetime.utcnow()})
                print(f"Updated timestamp for existing chat: {chat_id}")
            if not existing_chat:
                user_chat = UserChatList(
                    user_id=user_id,
                    chat_id=chat_id,
                    name=f"Suhbat: {req.query[:30]}...",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                await user_chat.insert()
                print(f"Added chat to user's chat list: {chat_id}")
            else:
                await existing_chat.set({"updated_at": datetime.utcnow()})
                print(f"Updated timestamp for existing chat: {chat_id}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
