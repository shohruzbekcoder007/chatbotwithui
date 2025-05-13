import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, ClassVar
from dotenv import load_dotenv
from functools import lru_cache
from typing import AsyncGenerator
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
_MODEL_SEMAPHORE = asyncio.Semaphore(10)  # Bir vaqtda 5 ta so'rovga ruxsat

class LangChainOllamaModel:
    """Ollama bilan LangChain integratsiyasi - ko'p foydalanuvchilar uchun optimallashtirilgan"""
    
    # Sinf darajasidagi o'zgaruvchilar
    _instances: ClassVar[Dict[str, 'LangChainOllamaModel']] = {}
    
    def __init__(self, 
                session_id: Optional[str] = None,
                model_name: str = "mistral-small:24b",
                base_url: str = "http://localhost:11434",
                temperature: float = 0.7 ,
                num_ctx: int = 2048,
                num_gpu: int = 1,
                gpu_layers: int = 100,
                kv_cache: bool = True, # buni xalaqit berishi mumkin
                num_thread: int = 32):
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
        self.gpu_layers = gpu_layers
        self.kv_cache = kv_cache
        
        # Model yaratish yoki cache dan olish
        self.model = self._get_model()
        
        # System prompts ro'yxati
        self.default_system_prompts = [
            "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan.",
            "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI.",
            "Let the information be based primarily on context and relegate additional answers to a secondary level.",
            "Do NOT integrate information that is not available in context. Kontextda mavjud bo'lmagan ma'lumotlarni qo‘shmaslik kerak.",
            "You must respond only in {language}. Ensure there are no spelling mistakes.",
            "Translate the answers into {language}.",
            "You should generate responses strictly based on the given prompt information without creating new content on your own.",
            "You are only allowed to answer questions related to the National Statistics Committee.",
            "The questions are within the scope of the estate and reports.",
            # "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)",
            "Each response must be formatted in HTML. Follow the guidelines below: Use <p> for text blocks, Use <strong> or <b> for important words, Use <ul> and <li> for lists, Use <code> for code snippets, Use <br> for line breaks within text, Every response should maintain semantic and visual clarity",
            "Integrate information that is not available in context. Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma",
            "Don't make up your own questions and answers, just use the information provided. O'zing savolni javobni to'qib chiqarma faqat berilgan ma'lumotlardan foydalan.",
            "Give a complete and accurate answer",
            "Don't add unrelated context",
            "Write the information as if you knew it in advance, don't imply that it was gathered from context.",
            # "If the answer is long, add a summary at the end of the answer using the format: '<br><p>  SUMMARY </p>'.",
            # "after answer, prefix one follow-up correctly short question with ‘Tavsiya qilingan savol:’, if the answer is a greeting, include nothing. ‘<br><p><b>Tavsiya qilingan keyingi savol:</b> Rasmiy statistika to‘g‘risidagi qonun nechta bo'limdan iborat?</p>’. answer with html tags",
            "If the answer is not clear or not answer, please clarify from the user. For Example: \"Savolingizni tushunmadim, Iltimos savolga aniqlik kiriting\"",
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
                    "num_thread": self.num_thread,
                    "gpu_layers": self.gpu_layers,
                    # "kv_cache": self.kv_cache,
                    "batch_size": 512,
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
    
    async def chat(self, context: str, query: str, language: str = "uz") -> str:
        """
        Modelga savol yuborish va javob olish
        
        Args:
            prompt (str): Savol matni
        
        Returns:
            str: Model javobi
        """
        # System va user promptlaridan xabarlar yaratish
        messages = self._create_messages(context, query, language)
        
        # Modelni chaqirish
        response = await self._invoke(messages)
        
        return response
    
    def _create_messages(self, system_prompts: str, user_prompt: str, language: str = "uz") -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        detect_lang = {
            "uz": "uzbek",
            "ru": "russian",
        }
        messages = []
        
        # System promptlarni qo'shish
        messages.append(SystemMessage(content="\n".join(self.default_system_prompts).format(language=detect_lang[language])))
        messages.append(SystemMessage(content=system_prompts))
        
        # User promptni qo'shish
        messages.append(HumanMessage(content=user_prompt))
        if language == "ru":
            messages.append(HumanMessage(content="Answer translate the result into Russian for me."))
        
        return messages
    
    async def _invoke(self, messages: List[BaseMessage]) -> str:
        """
        LangChain orqali modelni chaqirish (semaphore bilan cheklangan)
        """
        # for message in messages:
            # print(f"{self._get_message_type(message)}: {message.content}")
        # Global semaphore bilan cheklash
        async with _MODEL_SEMAPHORE:
            logger.info(f"Model chaqirilmoqda (session: {self.session_id})")
            
            try:
                # Ollama modelini chaqirish
                response = await asyncio.wait_for(self.model.ainvoke(messages), timeout=60.0)
                if isinstance(response, dict):
                    # Yangi LangChain versiyasi uchun
                    return response.get("content", str(response))
                # Eski versiya uchun
                return response.content
                
            except asyncio.TimeoutError:
                logger.warning("So‘rov muddati tugadi — model chaqiruvi bekor qilindi.")
                # GPU yukini tozalash yoki logga qo‘shish
                self._handle_timeout_cleanup()
                return "Modeldan javob olish vaqti tugadi. Iltimos, qaytadan urinib ko‘ring. Yoki savolni tushunarliroq qilib bering"

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

    async def chat_stream(self, context: str, query: str, language: str = "uz") -> AsyncGenerator[str, None]:
        """
        Javobni SSE orqali stream ko‘rinishda yuborish
        
        Args:
            context (str): System prompt
            query (str): User savoli
            language (str): Til
        
        Yields:
            str: Modeldan kelayotgan har bir token yoki parcha
        """
        messages = self._create_messages(context, query, language)
        async for chunk in self._stream_invoke(messages):
            yield chunk

    async def _stream_invoke(self, messages: List[BaseMessage]) -> AsyncGenerator[str, None]:
        """
        LangChain modelidan tokenlar ketma-ket stream tarzida olish
        """
        async with _MODEL_SEMAPHORE:
            try:
                logger.info(f"Model stream boshladi (session: {self.session_id})")
                async for chunk in self.model.astream(messages):
                    # LangChain chunk object: {'type': 'chat', 'message': AIMessage(content='...')}
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    elif isinstance(chunk, dict):
                        yield chunk.get("content", "")
                    else:
                        yield str(chunk)
            except asyncio.TimeoutError:
                logger.warning("Stream timeout.")
                self._handle_timeout_cleanup()
                yield "data: [TIMEOUT] Modeldan javob olish vaqti tugadi\n\n"
            except Exception as e:
                logger.error(f"Stream xatoligi: {str(e)}")
                yield f"data: [ERROR] {str(e)}\n\n"

    async def get_stream_suggestion_question(self, suggested_context: str, query: str, answer: str, language: str) -> AsyncGenerator[str, None]:
        """
        Stream tarzida tavsiya qilingan savolni real vaqtda generatsiya qilish (generate API bilan)

        Args:
            context (str): Savollar konteksti (faqat tanlash mumkin bo‘lgan savollar)
            query (str): Foydalanuvchi so‘rovi

        Yields:
            str: Har bir token yoki matn bo‘lagi
        """
        detect_lang = {
            "uz": "O'zbek",
            "ru": "Rus",
        }
        system_prompt = (
            "Siz tavsiya beruvchi yordamchisiz. Foydalanuvchining sorovi va unga berilgan javobga asoslanib, "
            "quyida berilgan kontekst savollari ichidan mantiqan eng yaqin bitta savolni tanlashingiz kerak.\n\n"
            "Faqat **bitta** savolni tanlang. Yangi savol o‘ylab topmang, faqat taqdim etilgan savollar ichidan tanlang.\n"
            "Sizga berilgan kontekstda mavjud bo‘lgan savollardan faqat 1 tasini tanlang. Faqat bitta savolni tanlang. Faqatgina savolning o'zini yozing, boshqa hech qanday ma'lumot qo'shib yozmang.\n\n"
            "Each response must be formatted in HTML (answer only <i> tag). Every response should maintain semantic and visual clarity\n"
            "Yangi Savol {language} tilida bo'lishi kerak.\n"
            "Agar foydalanuvchining so‘rovi noaniq bo‘lsa, uni aniqlashtirishni so‘rang.\n"
            "Agar foydalanuvchi savoli salomlashish yoki tanishish to'g'risida bo'lsa, statistika nizomi to'g'risidagi savolni yozing.\n"
        )

        full_prompt = (
            f"Taqdim etilgan savollar:\n{suggested_context}\n\n" 
            f"Foydalanuvchi sorovi:\n{query}\n\n"
            f"Yordamchi javobi:\n{answer}\n\n"
            f"Tavsiya qilinadigan keyingi savol: <b> </b>"
        )
        messages = [
            {"role": "system", "content": system_prompt.format(language=detect_lang[language])},
            {"role": "user", "content": full_prompt}
        ]
        if language == "ru":
            messages.append( {"role": "user", "content":"Answer translate the result into Russian for me."})
        

        async for chunk in self._stream_invoke(messages):
            yield chunk


# Factory funksiya - model obyektini olish
@lru_cache(maxsize=10)  # Eng ko'p 10 ta sessiya uchun cache
def get_model_instance(session_id: Optional[str] = None, model_name: str = "mistral-small3.1:24b", base_url: str = "http://localhost:11434") -> LangChainOllamaModel:
    return LangChainOllamaModel(session_id=session_id, model_name=model_name, base_url=base_url)

# Asosiy model obyekti (eski kod bilan moslik uchun)
model = get_model_instance()