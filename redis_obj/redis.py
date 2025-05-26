import os
import asyncio
import logging
import redis.asyncio as redis
from typing import Optional, Dict
from functools import lru_cache
import json
import datetime
import shutil

class RedisSession:
    """Redis'da foydalanuvchi sessiyasini saqlash"""
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600, use_file_fallback=True, pool_size=10):
        self.ttl = ttl
        self.file_fallback = use_file_fallback
        self.file_storage_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "session_data")
        self.pool_size = pool_size
        self.redis_lock = asyncio.Lock()
        
        # Fayl katalogini yaratish (agar mavjud bo'lmasa)
        if self.file_fallback and not os.path.exists(self.file_storage_path):
            try:
                os.makedirs(self.file_storage_path)
                print(f"File storage directory created at {self.file_storage_path}")
            except Exception as e:
                print(f"Failed to create file storage directory: {str(e)}")
        
        # Redis serverga ulanish
        try:
            self.redis_pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                max_connections=pool_size,
                decode_responses=True
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            # Test connection
            asyncio.run(self._test_connection())
            self.connected = True
            print("Successfully connected to Redis server")
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {str(e)}. Running with file-based storage.")
            self.connected = False
            
    async def _test_connection(self):
        """Redis ulanishini sinab ko'rish"""
        await self.redis_client.ping()
            
    def _get_file_path(self, key):
        """Sessiya kaliti uchun fayl yo'lini olish"""
        # Xavfsiz fayl nomi yaratish
        safe_key = key.replace(":", "_").replace("/", "_")
        return os.path.join(self.file_storage_path, f"{safe_key}.json")
            
    async def set_user_session(self, user_id, data):
        """Foydalanuvchi sessiyasini Redis'ga yozish"""
        if self.connected:
            try:
                async with self.redis_lock:
                    await self.redis_client.setex(f"session:{user_id}", self.ttl, json.dumps(data))
                return True
            except Exception as e:
                print(f"Warning: Redis connection failed while setting session: {str(e)}")
                
        # Redis serverga ulanib bo'lmaganda fayl tizimiga yozish
        if self.file_fallback:
            try:
                file_path = self._get_file_path(f"session:{user_id}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                print(f"Warning: File storage failed: {str(e)}")
        
        return False

    async def get_user_session(self, user_id) -> Optional[dict]:
        """Foydalanuvchi sessiyasini Redis'dan olish"""
        if self.connected:
            try:
                async with self.redis_lock:
                    data = await self.redis_client.get(f"session:{user_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Warning: Redis connection failed while getting session: {str(e)}")
                
        # Redis serverga ulanib bo'lmaganda fayl tizimidan o'qish
        if self.file_fallback:
            try:
                file_path = self._get_file_path(f"session:{user_id}")
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.loads(f.read())
            except Exception as e:
                print(f"Warning: File reading failed: {str(e)}")
        
        return None

    async def delete_user_session(self, user_id):
        """Foydalanuvchi sessiyasini Redis'dan o'chirish"""
        if self.connected:
            try:
                async with self.redis_lock:
                    await self.redis_client.delete(f"session:{user_id}")
            except Exception as e:
                print(f"Warning: Redis connection failed while deleting session: {str(e)}")

        # Redis serverga ulanib bo'lmaganda fayl tizimidan o'chirish
        if self.file_fallback:
            try:
                file_path = self._get_file_path(f"session:{user_id}")
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: File deletion failed: {str(e)}")

    async def delete_all_sessions(self):
        """Redis'da barcha sessiyalarni o'chirish"""
        if self.connected:
            try:
                async with self.redis_lock:
                    await self.redis_client.flushdb()
            except Exception as e:
                print(f"Warning: Redis connection failed while flushing database: {str(e)}")

        # Redis serverga ulanib bo'lmaganda barcha sessiya fayllarini o'chirish
        if self.file_fallback:
            try:
                shutil.rmtree(self.file_storage_path)
                os.makedirs(self.file_storage_path)
                print("All session files deleted")
            except Exception as e:
                print(f"Warning: Failed to delete session files: {str(e)}")

    async def set_question_session(self, user_id: str, chat_id: str, question: str):
        """
        Foydalanuvchi savoli va javobini Redis-ga saqlash
        
        Args:
            user_id: Foydalanuvchi ID si
            chat_id: Chat ID si
            question: Foydalanuvchi savoli
        """
        if not self.connected and not self.file_fallback:
            return
        try:
            # Sessiya kaliti uchun user_id va chat_id ni birlashtirish
            session_key = f"{user_id}:{chat_id}"
            session_data = await self.get_user_session(session_key) or {}
            
            # Oldingi savollar ro'yxatini olish yoki yangi ro'yxat yaratish
            chat_history = session_data.get("chat_history", [])
            
            # Yangi savol-javobni qo'shish
            chat_history.append({
                "question": question,
                "timestamp": str(datetime.datetime.now()),
                "chat_id": chat_id
            })
            
            # Ro'yxatni 10 ta so'nggi savol-javob bilan cheklash
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
            # Yangilangan ma'lumotlarni saqlash
            session_data["chat_history"] = chat_history
            await self.set_user_session(session_key, session_data)
            
            return True
        except Exception as e:
            print(f"Warning: Failed to save question and answer: {str(e)}")
            return False

    async def get_question_session(self, user_id: str, chat_id: str = None) -> list:
        """
        Foydalanuvchi savol-javoblar tarixini Redis-dan olish
        
        Args:
            user_id: Foydalanuvchi ID si
            chat_id: Chat ID si (ixtiyoriy)
            
        Returns:
            list: Savol-javoblar tarixi, [{"question": "...", "answer": "..."}, ...]
        """
        if not self.connected and not self.file_fallback:
            return []
        try:
            if chat_id:
                # Aniq chat uchun ma'lumotlarni olish
                session_key = f"{user_id}:{chat_id}"
                session_data = await self.get_user_session(session_key) or {}
                return session_data.get("chat_history", [])
            else:
                # Barcha chatlar uchun ma'lumotlarni yig'ib olish
                all_history = []
                # Redis-dan kalitlarni izlash
                if self.connected:
                    async with self.redis_lock:
                        keys = await self.redis_client.keys(f"session:{user_id}:*")
                    for key in keys:
                        session_data = await self.get_user_session(key.replace("session:", "")) or {}
                        all_history.extend(session_data.get("chat_history", []))
                elif self.file_fallback:
                    # Fayl tizimidan o'qish
                    for filename in os.listdir(self.file_storage_path):
                        if filename.startswith(f"session_{user_id}_") and filename.endswith(".json"):
                            file_path = os.path.join(self.file_storage_path, filename)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    session_data = json.load(f)
                                    all_history.extend(session_data.get("chat_history", []))
                            except Exception as e:
                                print(f"Warning: Failed to read file {filename}: {str(e)}")
                
                # Vaqt bo'yicha tartiblash
                all_history.sort(key=lambda x: x.get("timestamp", ""))
                return all_history
        except Exception as e:
            print(f"Warning: Failed to get question session: {str(e)}")
            return []
        
    async def set_suggestion_question(self, user_id: str, chat_id: str, suggested_question: str):
        """
        Tavsiya etilgan savollarni Redis-ga saqlash
    
        Args:
            user_id: Foydalanuvchi ID si
            chat_id: Chat ID si
            suggested_question: Tavsiya etilgan savol
        """
        if not self.connected and not self.file_fallback:
            print("Neither Redis nor file fallback is available")
            return False
            
        try:
            # Sessiya kaliti uchun user_id va chat_id ni birlashtirish
            session_key = f"{user_id}:{chat_id}"
            session_data = await self.get_user_session(session_key) or {}
        
            # Oldingi tavsiya etilgan savollar ro'yxatini olish yoki yangi ro'yxat yaratish
            suggested_questions = session_data.get("suggested_questions", [])
        
            # Yangi tavsiya etilgan savolni qo'shish
            if suggested_question and suggested_question not in suggested_questions:
                suggested_questions.append(suggested_question)
            
            # Ro'yxatni 10 ta so'nggi tavsiya etilgan savol bilan cheklash
            if len(suggested_questions) > 10:
                suggested_questions = suggested_questions[-10:]
                
            # Yangilangan ma'lumotlarni saqlash
            session_data["suggested_questions"] = suggested_questions
            print(f"Saving suggested question: {suggested_question}")
            return await self.set_user_session(session_key, session_data)
        except Exception as e:
            print(f"Warning: Failed to save suggested question: {str(e)}")
            return False

    async def get_suggested_questions(self, user_id: str, chat_id: str) -> list:
        """
        Tavsiya etilgan savollarni Redis-dan olish
    
        Args:
            user_id: Foydalanuvchi ID si
            chat_id: Chat ID si
        
        Returns:
            list: Tavsiya etilgan savollar ro'yxati
        """
        if not self.connected and not self.file_fallback:
            return []
        try:
            # Sessiya kaliti uchun user_id va chat_id ni birlashtirish
            session_key = f"{user_id}:{chat_id}"
            session_data = await self.get_user_session(session_key) or {}
        
            # Tavsiya etilgan savollar ro'yxatini olish
            return session_data.get("suggested_questions", [])
        except Exception as e:
            print(f"Warning: Failed to get suggested questions: {str(e)}")
            return []
        
# Redis sessiyasini yaratish
redis_session = RedisSession(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0)),
    ttl=int(os.getenv('REDIS_TTL', 3600)),
    use_file_fallback=True,
    pool_size=int(os.getenv('REDIS_POOL_SIZE', 10))
)

# Redis bilan ulanishni tekshirish
if not redis_session.connected:
    print("XATO: Redis serveriga ulanib bo'lmadi. Redis server ishga tushirilganligini tekshiring.")
else:
    print("INFO: Redis serveriga ulanish muvaffaqiyatli amalga oshirildi.")
