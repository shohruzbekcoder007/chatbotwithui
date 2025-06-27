from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from datetime import datetime, timezone
import time
from models.chat_message import ChatMessage
from models.user_chat_list import UserChatList
from additional.additional import change_translate, combine_text_arrays, get_docs_from_db, change_redis, is_russian, old_context, clean_html_tags
from models_llm.langchain_ollamaCustom import model as model_llm
from report_agent_executor import agent_stream_report
from retriever.langchain_chroma import questions_manager
from pydantic import BaseModel
from typing import Optional
from multi_agent_executor import combined_agent, agent_stream

# Router yaratish
router = APIRouter(prefix="", tags=["chat"])

# Request modelini yaratish
class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None
    device: Optional[str] = None  # Qo'shimcha maydon, agar kerak bo'lsa
    tool: Optional[str] = "all"  # Tool parametri, default "all"

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
        device = chat_request.device if chat_request.device else "web"
        tool = chat_request.tool if chat_request.tool else "all"
        # print(f"Using device: {device}")
        # print(f"Using tool: {tool}")

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
        await change_redis(user_id, chat_request.query, response_current)
        
        # Savol va javobni MongoDB ga saqlash (faqat autentifikatsiya qilingan foydalanuvchilar uchun)
        if user_id != "anonymous":
            chat_message = ChatMessage(
                user_id=user_id,
                chat_id=chat_id,
                message=chat_request.query,
                response=response_current,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            await chat_message.insert()
            print(f"Saved message to MongoDB with ID: {chat_message.id}")
            
            # UserChatList jadvaliga tekshirish va yozish
            existing_chat = await UserChatList.find_one(
                UserChatList.user_id == user_id,
                UserChatList.chat_id == chat_id
            )
            
            # Oldingi xabarlar sonini tekshirish
            previous_messages_count = await ChatMessage.find(
                ChatMessage.user_id == user_id,
                ChatMessage.chat_id == chat_id
            ).count() - 1  # Hozir saqlangan xabarni hisobga olmaslik
            
            if not existing_chat:
                # Birinchi savol - chat nomini to'liq saqlash
                chat_name = chat_request.query  # To'liq nomni saqlash
                new_chat = UserChatList(
                    user_id=user_id,
                    chat_id=chat_id,
                    name=chat_name,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                await new_chat.insert()
                print(f"Yangi chat yaratildi: {chat_id} with name: {chat_name}")
            else:
                # Agar bu birinchi xabar bo'lsa (oldingi xabarlar yo'q), chat nomini yangilash
                if previous_messages_count == 0:
                    chat_name = chat_request.query  # To'liq nomni saqlash
                    await existing_chat.set({
                        "name": chat_name,
                        "updated_at": datetime.now(timezone.utc)
                    })
                    print(f"Updated chat name to: {chat_name} for chat: {chat_id}")
                else:
                    # Faqat vaqtni yangilash, nomni o'zgartirmaslik
                    existing_chat.updated_at = datetime.now(timezone.utc)
                    await existing_chat.save()
                    print(f"Mavjud chat yangilandi: {chat_id}")
        else:
            print(f"Anonymous user, message not saved to     - Chat ID: {chat_id}")

        return {"response": response_current}

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}

@router.post("/chat/stream")
async def stream_chat(request: Request, req: ChatRequest):
    chat_id = req.chat_id
    # print(f"---------------- REQUEST: {req} ----------------")
    if not chat_id:
        return {"error": "Chat ID is missing. Please provide a valid chat ID."}

    question = req.query
    language = 'uz'
    device = req.device if req.device else "web"
    tool = req.tool if req.tool else "all"
    # print(f"\nUsing device: {device}")
    # print(f"Using question: {question}\n")
    print(f"Using tool: {tool}")

    user_id = request.state.user_id
    # print(f"Using user_id from request.state: {user_id}")
    suggestion_dict = {
        "web": {
            "uz": "\n <br><br><b> Tavsiya qilingan savol: </b> ",
            "ru": "\n <br><br><b> Предложенный вопрос: </b> "
        },
        "mobile": {
            "uz": "\n ** Tavsiya qilingan savol: ** ",
            "ru": "\n ** Предложенный вопрос: ** "
        },
    }

    if(is_russian(question)):
        question = await change_translate(question, "uz")
        language = 'ru'

    suggestion_text = suggestion_dict.get(device, {}).get(language, "")
    context_query = await old_context(user_id, question)

    print(context_query, "<< context_query")

    # print(suggestion_text, "<< suggestion_text")
    # print(device, "<< device")
    # print(language, "<< language")

    # Kontekstni tayyorlash
    relevant_docs = get_docs_from_db(question)
    relevant_docs_add = get_docs_from_db(context_query)

    docs = relevant_docs.get("documents", []) if isinstance(relevant_docs, dict) else []
    docs_add = relevant_docs_add.get("documents", []) if isinstance(relevant_docs_add, dict) else []

    unique_results = await combine_text_arrays(docs[0], docs_add[0])

    context = "\n- ".join(unique_results) if unique_results else ""

    # print(f"Context: ============================ \n {context}")

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
        # async for token in model_llm.chat_stream(context=context, query=context_query+"\n"+question, language=language, device=device):
        #     response_current += token
        #     yield f"{token}"

        if tool == "classifications":
            async for token in agent_stream(query=context_query+"\n"+question):
                response_current += token
                yield f"{token}"

        if tool == "report":
            async for token in agent_stream_report(query=context_query+"\n"+question):
                response_current += token
                yield f"{token}"

        if tool == "all":
            async for token in model_llm.chat_stream(context=context, query=context_query+"\n"+question, language=language, device=device):
                response_current += token
                yield f"{token}"
        
        # yield suggestion_text
        
        questions_manager_questions = questions_manager.search_documents(response_current, 10)
        suggested_question = ""
        if False: # questions_manager_questions:
            sq_docs = questions_manager_questions.get("documents", []) if isinstance(questions_manager_questions, dict) else []
            # print(sq_docs, "<<- sq_docs")
            question_docs = []
            previous_questions.add(req.query.strip().lower())
            # print(f"\n\nOldingi savollar: {list(previous_questions)}")
            for doc in sq_docs[0]:
                # Tavsiya qilingan savollarni oldingi savollar bilan solishtirish
                # print(f"\n - Previous question: {doc.lower()} - {doc.lower() in list(previous_questions)}")
                if doc.lower() not in list(previous_questions):
                    question_docs.append(doc)
                    
            suggestion_context = "\n- ".join(question_docs) if question_docs else ""
            # print(f"\n\nSuggestion question context: {suggestion_context}")

            # Stream orqali tavsiya qilingan savollarni olish
            async for token in model_llm.get_stream_suggestion_question(suggestion_context, question, response_current, language, device=device):
                suggested_question += token
                # yield f"{token}\n\n"  # Un-commenting this line to yield tokens

            suggestion_result = suggestion_text + suggested_question

            for token in suggestion_result:
                time.sleep(0.006)  # Tokenlarni chiqarish uchun kutish
                yield f"{token}"
        
        # MongoDB ga saqlash
        chat_message = ChatMessage(
            user_id=user_id,
            chat_id=chat_id,
            message=req.query,
            response=response_current,
            suggestion_question=clean_html_tags(suggested_question),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
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
                # Birinchi savol - chat nomini to'liq saqlash
                chat_name = req.query  # To'liq nomni saqlash
                user_chat = UserChatList(
                    user_id=user_id,
                    chat_id=chat_id,
                    name=chat_name,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                await user_chat.insert()
                print(f"Added chat to user's chat list: {chat_id} with name: {chat_name}")
            else:
                # Birinchi xabar emasligini tekshirish (oldingi xabarlar soni)
                message_count = len(previous_messages)
                
                # Agar bu birinchi xabar bo'lsa (oldingi xabarlar yo'q), chat nomini yangilash
                if message_count == 0:
                    chat_name = req.query  # To'liq nomni saqlash
                    await existing_chat.set({
                        "name": chat_name,
                        "updated_at": datetime.now(timezone.utc)
                    })
                    print(f"Updated chat name to: {chat_name} for chat: {chat_id}")
                else:
                    # Faqat vaqtni yangilash, nomni o'zgartirmaslik
                    await existing_chat.set({"updated_at": datetime.now(timezone.utc)})
                    print(f"Updated timestamp for existing chat: {chat_id}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")