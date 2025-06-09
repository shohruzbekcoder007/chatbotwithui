from fastapi import APIRouter, Request, HTTPException, status
from datetime import datetime
import uuid
from models.user_chat_list import UserChatList
from models.chat_message import ChatMessage

# Router yaratish
router = APIRouter(prefix="/api", tags=["chat"])

@router.get("/user-chats")
async def get_user_chats(request: Request, limit: int = 10, offset: int = 0):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous user bo'lsa, xato qaytaramiz
        if user_id == "anonymous":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Umumiy chat sonini olish (pagination uchun)
        total_chats = await UserChatList.find(UserChatList.user_id == user_id).count()
        
        # Foydalanuvchining suhbatlarini pagination bilan olish
        user_chats = await UserChatList.find(
            UserChatList.user_id == user_id
        ).sort("-updated_at").skip(offset).limit(limit).to_list()  # Eng so'nggi yangilangan birinchi
        
        # Formatlash
        formatted_chats = []
        for chat in user_chats:
            formatted_chats.append({
                "chat_id": chat.chat_id,
                "name": chat.name,
                "created_at": chat.created_at.isoformat(),
                "updated_at": chat.updated_at.isoformat()
            })
        
        return {
            "success": True, 
            "chats": formatted_chats,
            "pagination": {
                "total": total_chats,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_chats
            }
        }
    except HTTPException as e:
        # HTTPException qayta ko'taramiz
        raise e
    except Exception as e:
        print(f"Error retrieving user chats: {str(e)}")
        return {"success": False, "error": str(e)}

@router.get("/chat/{chat_id}")
async def get_chat(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Anonymous users cannot access chat details"
            )
        
        # Chat ma'lumotini olish
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if not chat:
            return {"success": False, "error": "Chat not found"}
        
        # Chat xabarlarini olish
        messages = await ChatMessage.find(
            ChatMessage.chat_id == chat_id,
            ChatMessage.user_id == user_id
        ).sort("+created_at").to_list()
        
        # Natijani formatlash
        result = {
            "chat_id": chat.chat_id,
            "name": chat.name,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat(),
            "messages": []
        }
        
        # Xabarlarni qo'shish
        for msg in messages:
            result["messages"].append({
                "id": str(msg.id),
                "message": msg.message,
                "response": msg.response,
                "created_at": msg.created_at.isoformat(),
                "updated_at": msg.updated_at.isoformat()
            })
        
        return {"success": True, "data": result}
    except HTTPException as e:
        # HTTPException qayta ko'taramiz
        raise e
    except Exception as e:
        print(f"Error retrieving chat: {str(e)}")
        return {"success": False, "error": str(e)}

@router.post("/chat/{chat_id}/rename")
async def rename_chat(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Anonymous users cannot rename chats"
            )
        
        # JSON ma'lumotlarini olish
        data = await request.json()
        new_name = data.get("name", "Yangi suhbat")
        
        # UserChatList jadvalidagi chat nomini yangilash
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if chat:
            await chat.set({
                "name": new_name,
                "updated_at": datetime.utcnow()
            })
            return {"success": True, "message": "Chat renamed successfully"}
        else:
            return {"success": False, "error": "Chat not found"}
    except HTTPException as e:
        # HTTPException qayta ko'taramiz
        raise e
    except Exception as e:
        print(f"Error renaming chat: {str(e)}")
        return {"success": False, "error": str(e)}

@router.delete("/chat/{chat_id}")
async def delete_chat(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Anonymous users cannot delete chats"
            )
        
        # UserChatList jadvalidan o'chirish
        chat = await UserChatList.find_one(
            UserChatList.user_id == user_id,
            UserChatList.chat_id == chat_id
        )
        
        if chat:
            await chat.delete()
            
            # ChatMessage jadvalidagi xabarlarni ham o'chirish
            await ChatMessage.find(
                ChatMessage.user_id == user_id,
                ChatMessage.chat_id == chat_id
            ).delete_many()
            
            return {"success": True, "message": "Chat and all messages deleted"}
        else:
            return {"success": False, "error": "Chat not found"}
    except HTTPException as e:
        # HTTPException qayta ko'taramiz
        raise e
    except Exception as e:
        print(f"Error deleting chat: {str(e)}")
        return {"success": False, "error": str(e)}

@router.post("/chat")
async def create_chat(request: Request):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Anonymous foydalanuvchilar uchun xatolik
        if user_id == "anonymous":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Anonymous users cannot create chats"
            )
        
        # Yangi chat ID yaratish
        new_chat_id = str(uuid.uuid4())
        
        # UserChatList ga yangi chat qo'shish
        chat = UserChatList(
            user_id=user_id,
            chat_id=new_chat_id,
            name="Yangi suhbat",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await chat.insert()
        
        return {"success": True, "chat_id": new_chat_id}
    except HTTPException as e:
        # HTTPException qayta ko'taramiz
        raise e
    except Exception as e:
        print(f"Error creating chat: {str(e)}")
        return {"success": False, "error": str(e)}

@router.get("/chat-history/{chat_id}")
async def get_chat_history(request: Request, chat_id: str):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Faqat o'z suhbatlarini ko'rish imkonini berish
        messages = await ChatMessage.find(
            ChatMessage.chat_id == chat_id,
            ChatMessage.user_id == user_id
        ).sort("+created_at").to_list()
        
        # Xabarlarni formatlash
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "id": str(msg.id),
                "message": msg.message,
                "response": msg.response,
                "created_at": msg.created_at.isoformat(),
                "updated_at": msg.updated_at.isoformat()
            })
        
        return {"success": True, "messages": formatted_messages}
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return {"success": False, "error": str(e)}

@router.put("/chat-message/{message_id}")
async def update_chat_message(request: Request, message_id: str, update_data: dict):
    try:
        # user_id ni olish
        user_id = request.state.user_id
        
        # Xabarni topish
        message = await ChatMessage.get(message_id)
        
        # Xabar topilmasa yoki boshqa foydalanuvchiga tegishli bo'lsa, xatolik
        if not message or message.user_id != user_id:
            return {"success": False, "error": "Message not found or access denied"}
        
        # Xabarni yangilash
        update_fields = {}
        
        if "message" in update_data:
            update_fields["message"] = update_data["message"]
            
        if "response" in update_data:
            update_fields["response"] = update_data["response"]
        
        # updated_at ni har doim yangilash
        update_fields["updated_at"] = datetime.utcnow()
        
        # Yangilash
        await message.set({**update_fields})
        
        return {"success": True, "message": "Message updated successfully"}
    except Exception as e:
        print(f"Error updating chat message: {str(e)}")
        return {"success": False, "error": str(e)}
