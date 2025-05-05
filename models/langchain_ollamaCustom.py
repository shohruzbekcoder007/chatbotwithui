import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, ClassVar
from dotenv import load_dotenv
from functools import lru_cache

# LangChain importlari
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage, 
    SystemMessage, 
    BaseMessage
)
from langchain_community.chat_models import ChatOllama

# .env faylidan konfiguratsiyani o'qish
load_dotenv()

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model va pipeline obyektlarini saqlash uchun cache
_MODEL_CACHE = {}

# Global semaphore - bir vaqtda nechta so'rov bajarilishi mumkinligini cheklaydi
_MODEL_SEMAPHORE = asyncio.Semaphore(5)  # Bir vaqtda 5 ta so'rovga ruxsat

class LangChainOllamaModel:
    """Ollama bilan LangChain integratsiyasi - ko'p foydalanuvchilar uchun optimallashtirilgan"""
    
    # Sinf darajasidagi o'zgaruvchilar
    _instances: ClassVar[Dict[str, 'LangChainOllamaModel']] = {}
    
    def __init__(self, 
                session_id: Optional[str] = None,
                model_name: str = "mistral-small:24b",
                base_url: str = "http://172.16.8.39:11434",
                temperature: float = 0.7 ,
                num_ctx: int = 4096,
                num_gpu: int = 1,
                num_thread: int = 8):
        """
        Ollama modelini LangChain bilan ishlatish uchun klass.
        
        Args:
            session_id: Foydalanuvchi sessiyasi ID si
            model_name: Ollama model nomi
            base_url: Ollama server URL
            temperature: Generatsiya uchun temperature
            num_ctx: Kontekst uzunligi
            num_gpu: Ishlatilishi kerak bo'lgan GPU soni
            num_thread: Ishlatilishi kerak bo'lgan CPU thread soni
        """
        self.session_id = session_id or "default"
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        
        # Model yaratish yoki cache dan olish
        self.model = self._get_model()
        
        # System prompts ro'yxati
        self.default_system_prompts = [
            "Let the information be based primarily on context and relegate additional answers to a secondary level.",
            "Do NOT integrate information that is not available in context. Kontextda mavjud bo'lmagan ma'lumotlarni qo‘shmaslik kerak."
            "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI.",
            "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan.",
            "You must respond only in Uzbek. Ensure there are no spelling mistakes.",
            "You should generate responses strictly based on the given prompt information without creating new content on your own.",
            "You are only allowed to answer questions related to the National Statistics Committee.",
            "The questions are within the scope of the estate and reports.",
            "don't add unrelated context",
            # "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)",
            "Each response must be formatted in HTML. Follow the guidelines below: Use <p> for text blocks, Use <strong> or <b> for important words, Use <ul> and <li> for lists, Use <code> for code snippets, Use <br> for line breaks within text, Every response should maintain semantic and visual clarity"
            "Integrate information that is not available in context. Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma",
            "Don't make up your own questions and answers, just use the information provided. O'zing savolni javobni to'qib chiqarma faqat berilgan ma'lumotlardan foydalan.",
            "Give a complete and accurate answer",
            "If the answer is not clear, add the sentence \"Iltimos, savolga aniqlik kiriting\""
        ]
    
    def _get_model(self):
        """Model olish (cache dan yoki yangi yaratish)"""
        global _MODEL_CACHE
        
        cache_key = f"{self.model_name}_{self.base_url}"
        
        if cache_key not in _MODEL_CACHE:
            logger.info(f"Ollama model yaratilmoqda: {self.model_name}")
            model = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                context_window=self.num_ctx,
                extra_model_kwargs={
                    "num_gpu": self.num_gpu,
                    "num_thread": self.num_thread
                }
            )
            _MODEL_CACHE[cache_key] = model
        
        return _MODEL_CACHE[cache_key]
    
    def _get_message_type(self, message: BaseMessage) -> str:
        """Message turini aniqlash"""
        if isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
        else:
            return "unknown"
    
    def _create_message_from_dict(self, message_dict: Dict[str, Any]) -> BaseMessage:
        """Dictionary ma'lumotlaridan message obyekti yaratish"""
        message_type = message_dict.get("type", "")
        content = message_dict.get("content", "")
        
        if message_type == "system":
            return SystemMessage(content=content)
        elif message_type == "human":
            return HumanMessage(content=content)
        elif message_type == "ai":
            return AIMessage(content=content)
        else:
            # Noma'lum turlar uchun default HumanMessage
            return HumanMessage(content=content)
    
    async def chat(self, context: str, query: str) -> str:
        """
        Modelga savol yuborish va javob olish
        
        Args:
            prompt (str): Savol matni
        
        Returns:
            str: Model javobi
        """
        # System va user promptlaridan xabarlar yaratish
        messages = self._create_messages(context, query)
        
        # Modelni chaqirish
        response = await self._invoke(messages)
        
        return response
    
    def _create_messages(self, system_prompts: str, user_prompt: str) -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        messages = []
        
        # System promptlarni qo'shish
        # for prompt in system_prompts:
        messages.append(SystemMessage(content="\n".join(self.default_system_prompts)))
        messages.append(SystemMessage(content=system_prompts))
        
        # User promptni qo'shish
        messages.append(HumanMessage(content=user_prompt))
        
        return messages
    
    async def _invoke(self, messages: List[BaseMessage]) -> str:
        """
        LangChain orqali modelni chaqirish (semaphore bilan cheklangan)
        """
        # Global semaphore bilan cheklash
        async with _MODEL_SEMAPHORE:
            logger.info(f"Model chaqirilmoqda (session: {self.session_id})")
            
            try:
                # Ollama modelini chaqirish
                response = await asyncio.wait_for(self.model.ainvoke(messages), timeout=15.0)
                if isinstance(response, dict):
                    # Yangi LangChain versiyasi uchun
                    return response.get("content", str(response))
                # Eski versiya uchun
                return response.content
                
            except asyncio.TimeoutError:
                logger.warning("So‘rov muddati tugadi — model chaqiruvi bekor qilindi.")
                # GPU yukini tozalash yoki logga qo‘shish
                self._handle_timeout_cleanup()
                return "Xatolik: Modeldan javob olish muddati tugadi. Iltimos, keyinroq urinib ko‘ring."

            except Exception as e:
                logger.error(f"Model chaqirishda xatolik: {str(e)}")
                return f"Xatolik yuz berdi: {str(e)}"

    def _handle_timeout_cleanup(self):
        """
        Timeoutdan keyin aynan shu sessiyaga tegishli protsessni tozalash.
        """
        import psutil
        logger.info(f"Timeoutdan keyingi tozalash (session: {self.session_id})")
        print("Timeoutdan keyin tozalash jarayoni...")

        try:
            current_pid = os.getpid()  # Joriy PID
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Faqat tegishli protsesslarni qidiring
                    if proc.info['cmdline'] and "ollama" in " ".join(proc.info['cmdline']):
                        if str(self.session_id) in " ".join(proc.info['cmdline']):
                            logger.info(f"Sessionga tegishli protsess topildi: PID={proc.pid}")
                            proc.terminate()
                            logger.info(f"Protsess {proc.pid} to‘xtatildi.")
                            return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            logger.warning("Sessionga tegishli protsess topilmadi.")
        except Exception as e:
            logger.error(f"Timeout tozalashda xatolik: {str(e)}")

    async def logical_context(self, text: str) -> str:
        """
        Berilgan matnni mantiqiy qayta ishlash
        """
        system_prompt = """
        Sen mantiqiy kontekst yaratuvchi yordamchisan. Berilgan mavzu bo'yicha mantiqiy va faktlarga asoslangan 
        ma'lumotlarni tuzib berishing kerak. Faqat haqiqiy faktlarni ishlatib, barcha ma'lumotlarni aniq va qisqa shaklda yozib ber.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Quyidagi mavzu haqida mantiqiy kontekst yaratib ber: {text}")
        ]
        
        return await self._invoke(messages)
    
    async def generate_questions(self, question: str) -> List[str]:
        """
        Berilgan savolga asoslanib 3 ta sinonim savol yaratish
        """
        system_prompt = """
        Sen savol generatsiya qiluvchi yordamchisan. Berilgan savolga o'xshash, lekin boshqacha so'z birikmalaridan 
        foydalangan 3 ta savol yaratishing kerak. Savollar original savol bilan bir xil ma'noni anglatishi, 
        lekin boshqacha so'zlar bilan ifodalanishi kerak.
        
        Faqat 3 ta savolni qaytarish, har bir savolni alohida qatorda.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original savol: {question}")
        ]
        
        response = await self._invoke(messages)
        
        # Savollarni qatorlarga bo'lish
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        
        # Faqat 3 ta savolni qaytarish
        return questions[:3]
    
    async def rewrite_query(self, user_query: str, chat_history: str = "") -> str:
        """
        Foydalanuvchi so'rovini qayta yozish
        
        Args:
            user_query: Foydalanuvchi so'rovi
            chat_history: Chat tarixi (ixtiyoriy)
            
        Returns:
            str: Qayta yozilgan so'rov
        """
        system_prompt = """
        Sen so'rovlarni qayta yozish uchun yordamchisan. Berilgan chat tarixini hisobga olib, 
        foydalanuvchi so'rovini kontekstga mos ravishda qayta yozishing kerak.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat tarixi:\n{chat_history}\n\nAsl so'rov: {user_query}\n\nSo'rovni qayta yozing:")
        ]
        
        try:
            response = await self._invoke(messages)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"So'rovni qayta yozishda xatolik: {str(e)}")
            return user_query  # Xatolik bo'lsa asl so'rovni qaytarish
    
    def count_tokens(self, text: str) -> int:
        """
        Matndagi tokenlar sonini taxminiy hisoblash.
        
        Args:
            text (str): Token soni hisoblanadigan matn
            
        Returns:
            int: Taxminiy token soni
        """
        if not text:
            return 0
            
        # Oddiy tokenlashtirish qoidalari
        # 1. O'rtacha 1 token = 4 belgi ingliz tilida
        # 2. Har bir so'z taxminan 1.3 token
        # 3. Har bir tinish belgisi alohida token
        
        # So'zlar soni (bo'shliqlar bo'yicha ajratish)
        words = len(text.split())
        
        # Tinish belgilari soni
        punctuation = sum(1 for char in text if char in '.,;:!?()-"\'[]{}')
        
        # Umumiy belgilar soni
        chars = len(text)
        
        # Ikki usul bilan taxminlash va o'rtachasini olish
        estimate1 = chars / 4  # Har 4 belgi uchun 1 token
        estimate2 = words * 1.3 + punctuation  # Har bir so'z 1.3 token + tinish belgilari
        
        # Ikki usul o'rtachasi
        result = int((estimate1 + estimate2) / 2)
        
        return max(1, result)


# Factory funksiya - model obyektini olish
@lru_cache(maxsize=10)  # Eng ko'p 10 ta sessiya uchun cache
def get_model_instance(session_id: Optional[str] = None, model_name: str = "mistral-small:24b", base_url: str = "http://localhost:11434") -> LangChainOllamaModel:
    return LangChainOllamaModel(session_id=session_id, model_name=model_name, base_url=base_url)

# Asosiy model obyekti (eski kod bilan moslik uchun)
model = get_model_instance()