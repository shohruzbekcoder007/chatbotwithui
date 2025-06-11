import os
import asyncio
import logging
import torch
from typing import List, Dict, Any, Optional, ClassVar, AsyncGenerator
from dotenv import load_dotenv
from functools import lru_cache
from langchain_core.messages import (
    AIMessage,
    HumanMessage, 
    SystemMessage, 
    BaseMessage
)
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from tools_llm.country.country_tool import CountryTool
from tools_llm.dbibt.dbibt_tool import DBIBTTool
from tools_llm.soato.soato_tool import SoatoTool
from tools_llm.nation.nation_tool import NationTool
from tools_llm.ckp.ckp_tool_simple import CkpTool
from retriever.langchain_chroma import CustomEmbeddingFunction
from tools_llm.thsh.thsh_tool import THSHTool
from tools_llm.tif_tn.tif_tn_tool import TIFTNTool

# .env faylidan konfiguratsiyani o'qish
load_dotenv()

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model va pipeline obyektlarini saqlash uchun cache
_MODEL_CACHE = {}

# Global semaphore - bir vaqtda nechta so'rov bajarilishi mumkinligini cheklaydi
_MODEL_SEMAPHORE = asyncio.Semaphore(10)  # Bir vaqtda 10 ta so'rovga ruxsat

class LangChainOllamaModel:
    """Ollama bilan LangChain integratsiyasi - ko'p foydalanuvchilar uchun optimallashtirilgan"""
    
    # Sinf darajasidagi o'zgaruvchilar
    _instances: ClassVar[Dict[str, 'LangChainOllamaModel']] = {}
    
    def __init__(self, 
                session_id: Optional[str] = None,
                model_name: str = "mistral-small:24b",
                base_url: str = "http://localhost:11434",
                temperature: float = 0.7,
                num_ctx: int = 2048,
                num_gpu: int = 1,
                gpu_layers: int = 100,
                kv_cache: bool = True,
                num_thread: int = 32,
                use_agent: bool = True):
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
            use_agent: Agent ishlatish yoki yo'q
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
        self.use_agent = use_agent
        self.agent = None
        
        # System prompts ro'yxati
        self.default_system_prompts = [
            "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan.",
            "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI.",
            "Let the information be based primarily on context and relegate additional answers to a secondary level.",
            "Do NOT integrate information that is not available in context. Kontextda mavjud bo'lmagan ma'lumotlarni qo'shmaslik kerak.",
            "You must respond only in {language}. Ensure there are no spelling mistakes.",
            "Translate the answers into {language}.",
            "You should generate responses strictly based on the given prompt information without creating new content on your own.",
            "You are only allowed to answer questions related to the National Statistics Committee.",
            "The questions are within the scope of the estate and reports.",
            "Don't make up your own questions and answers, just use the information provided. O'zing savolni javobni to'qib chiqarma faqat berilgan ma'lumotlardan foydalan.",
            "Give a complete and accurate answer.", 
            "Don't add unrelated context.",
            "Write the information as if you knew it in advance, don't imply that it was gathered from context.",
            "Faqat javobni yozing. Javobdan oldin yoki keyin hech qanday boshqa ma'lumot qo'shmang. Gaplarni bir biriga ulab yozgin.",
            "If the answer is not clear or not answer, please clarify from the user. For Example: \"Savolingizni tushunmadim, Iltimos savolga aniqlik kiriting\".",
            "Only use the parts of the context that are directly relevant to the user's question. Ignore all other context, even if it is statistically related. Use only what directly answers the question.",
            "Savolga javob berishda faqat kontekstga asoslaning. Agar kontekstda javob bo'lmasa, \"Savolingizni tushunmadim, aniqroq qilib savol bering\" deb yozing.",
            "If the question is about date, time, day of week, or month name, use the appropriate tool to get the current information."
        ]

        self.system_html_prompts = [
            "Each response must be formatted in HTML. Follow the guidelines below: Use <p> for text blocks, Use <strong> or <b> for important words, Use <ul> and <li> for lists, Use <code> for code snippets, Use <br> for line breaks within text, Every response should maintain semantic and visual clarity.",
        ]
        self.system_markdown_prompts = [
            "Each response must be formatted in Markdown. Follow the guidelines below: Use ` for inline code, Use ``` for code blocks, Use ** or __ for bold text, Use * or _ for italic text, Use > for blockquotes, Use - or * for bullet points, Every response should maintain semantic and visual clarity.",
        ]
        
        # Agent-related initialization
        if self.use_agent:
            self._initialize_agent()
        
        # Model yaratish yoki cache dan olish
        self.model = self._get_model()
    
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
                    "batch_size": 512,
                }
            )
            _MODEL_CACHE[cache_key] = model
        
        return _MODEL_CACHE[cache_key]
    
    def _initialize_agent(self):
        """
        Agent va uning toollarini ishga tushirish
        """
        try:
            # Model yaratish yoki cache dan olish
            self.model = self._get_model()
            
            # Bitta embedding modelini yaratish, barcha toollar uchun
            shared_embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
            
            # Toollarni yaratish - yangi improved SOATO tool
            logger.info("Improved SOATO tool yaratilmoqda...")
            soato_tool = SoatoTool(
                soato_file_path="tools_llm/soato/soato.json", 
                use_embeddings=True, 
                embedding_model=shared_embedding_model
            )
            
            nation_tool = NationTool("tools_llm/nation/nation_data.json", use_embeddings=True, embedding_model=shared_embedding_model)
            ckp_tool = CkpTool("tools_llm/ckp/ckp.json", use_embeddings=True, embedding_model=shared_embedding_model)
            dbibt_tool = DBIBTTool("tools_llm/dbibt/dbibt.json", use_embeddings=True, embedding_model=shared_embedding_model)
            country_tool = CountryTool("tools_llm/country/country.json", use_embeddings=True, embedding_model=shared_embedding_model)
            thsh_tool = THSHTool("tools_llm/thsh/thsh.json", use_embeddings=True, embedding_model=shared_embedding_model)
            tif_tn_tool = TIFTNTool("tools_llm/tif_tn/tif_tn.json", use_embeddings=True, embedding_model=shared_embedding_model)
            
            # Embedding ma'lumotlarini tayyorlash
            logger.info("Barcha toollar uchun embedding ma'lumotlari tayyorlanmoqda...")
            try:
                # SOATO tool embedding ma'lumotlarini tayyorlash
                if hasattr(soato_tool, '_prepare_embedding_data'):
                    soato_tool._prepare_embedding_data()
                    logger.info("SOATO tool embedding ma'lumotlari tayyor")
                
                # Boshqa toollar uchun embedding tayyorlash
                if hasattr(nation_tool, '_prepare_embedding_data'):
                    nation_tool._prepare_embedding_data()
                if hasattr(ckp_tool, '_prepare_embedding_data'):
                    ckp_tool._prepare_embedding_data()
                if hasattr(dbibt_tool, '_prepare_embedding_data'):
                    dbibt_tool._prepare_embedding_data()
                if hasattr(country_tool, '_prepare_embedding_data'):
                    country_tool._prepare_embedding_data()
                if hasattr(thsh_tool, '_prepare_embedding_data'):
                    thsh_tool._prepare_embedding_data()
                if hasattr(tif_tn_tool, '_prepare_embedding_data'):
                    tif_tn_tool._prepare_embedding_data()
                logger.info("Barcha toollar uchun embedding ma'lumotlari tayyor")
                
            except Exception as e:
                logger.warning(f"Embedding ma'lumotlarini tayyorlashda ogohlantirish: {e}")
            
            # Agentni yaratish - mavjud modeldan foydalanish
            self.agent = initialize_agent(
                tools=[soato_tool, nation_tool, ckp_tool, dbibt_tool, country_tool, thsh_tool, tif_tn_tool],
                llm=self.model,  # Yangi model yaratish o'rniga mavjud modeldan foydalanish
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=True,
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
                system_message=SystemMessage(content="""Siz O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun maxsus agentsiz. 
                    Sizda quyidagi toollar mavjud:
                    
                    1. soato_tool - O'zbekiston Respublikasining ma'muriy-hududiy birliklari (SOATO/MHOBIT) ma'lumotlarini qidirish uchun mo'ljallangan professional vosita. Bu tool LangChain best practices asosida yaratilgan va quyidagi imkoniyatlarni taqdim etadi:
                       - Viloyatlar, tumanlar, shaharlar va aholi punktlarning soato yoki mhobt kodlari va nomlari
                       - SOATO/MHOBIT kodlari bo'yicha aniq qidiruv
                       - Ma'muriy birliklarning ierarxik tuzilishi
                       - Viloyatlar, tumanlar va aholi punktlari soni haqida ma'lumotlar
                       Misol so'rovlar: "Toshkent viloyati tumanlari mhobt kodlari", "Samarqand viloyati soato si qanday", "1710 soato kodi", "Buxoro viloyati tumanlari soni", "Andijon ma'muriy markazi"
                    
                    2. nation_tool - O'zbekiston Respublikasi millat klassifikatori ma'lumotlarini qidirish uchun tool. Bu tool orqali millat kodi yoki millat nomi bo"yicha qidiruv qilish mumkin. Masalan: "01" (o'zbek), "05" (rus), yoki "tojik" kabi so"rovlar bilan qidiruv qilish mumkin. Tool millat kodi, nomi va boshqa tegishli ma'lumotlarni qaytaradi.
                    
                    3. ckp_tool - MST/CKP (Mahsulotlarning tasniflagichi) ma'lumotlarini qidirish va tahlil qilish uchun mo'ljallangan vosita. Bu tool orqali mahsulotlar kodlari, nomlari va tasniflarini izlash, ularning ma'lumotlarini ko"rish va tahlil qilish mumkin. Qidiruv so'z, kod yoki tasnif bo"yicha amalga oshirilishi mumkin.
                    
                    4. dbibt_tool - O'zbekiston Respublikasi Davlat va xo'jalik boshqaruvi idoralarini belgilash tizimi (DBIBT) ma'lumotlarini qidirish uchun tool. Bu tool orqali DBIBT kodi, tashkilot nomi, OKPO/KTUT yoki STIR/INN raqami bo"yicha qidiruv qilish mumkin. Masalan: "08824", "Vazirlar Mahkamasi", yoki "07474" kabi so"rovlar bilan qidiruv qilish mumkin.

                    5. country_tool - Davlatlar ma'lumotlarini qidirish uchun mo'ljallangan vosita. Bu tool orqali davlatlarning qisqa nomi, to'liq nomi, harf kodi va raqamli kodi bo'yicha qidiruv qilish mumkin. Misol uchun: "AQSH", "Rossiya", "UZ", "398" (Qozog'iston raqamli kodi) kabi so'rovlar orqali ma'lumotlarni izlash mumkin. Natijalar davlat kodi, qisqa nomi, to'liq nomi va kodlari bilan qaytariladi.

                    6. thsh_tool - Tashkiliy-huquqiy shakllar (THSH) ma'lumotlarini qidirish uchun mo'ljallangan vosita. Bu tool orqali tashkiliy-huquqiy shakllarning kodi, nomi, qisqa nomi va boshqa ma'lumotlar bo'yicha qidiruv qilish mumkin. Misol uchun: "110" (Xususiy korxona), "Aksiyadorlik jamiyati", "MJ" (Mas'uliyati cheklangan jamiyat) kabi so'rovlar orqali ma'lumotlarni izlash mumkin.
                    
                    7. tif_tn_tool - Tashqi iqtisodiy faoliyat tovarlar nomenklaturasi (TIF-TN). Bu tool orqali tashqi iqtisodiy faoliyat tovarlar nomenklaturasi bo'yicha qidiruv qilish mumkin. Misol uchun: '0102310000' (Tirik yirik shoxli qoramol), '0101210000' (Tirik otlar, eshaklar, xachirlar va xo' tiklar), '0102399000' (Tirik yirik shoxli qoramol) kabi so'rovlar orqali ma'lumotlarni izlash mumkin.
                    
                    Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning. 
                    Agar foydalanuvchi SOATO/MHOBIT (viloyat, tuman, shahar), millat, MST, davlat ma'lumotlari yoki tashkiliy-huquqiy shakllar haqida so'rasa, tegishli toolni chaqiring.
                    Toollarni chaqirish uchun Action formatidan foydalaning.
                    JUDA MUHIM QOIDALAR: 
                    1. Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning
                    2. Tool ishlatganingizdan keyin MAJBURIY ravishda Final Answer bering
                    
                     """),
                max_iterations=3,  # Maksimal iteratsiya sonini cheklash
                early_stopping_method="generate"  # Erta to'xtatish usuli
            )
            logger.info(f"Agent muvaffaqiyatli yaratildi (session: {self.session_id}) - Improved SOATO tool bilan")
        except Exception as e:
            logger.error(f"Agent yaratishda xatolik: {str(e)}")
            self.agent = None
            
    def _is_agent_query(self, query: str) -> bool:
        """
        So'rov agentga tegishli yoki yo'qligini aniqlash
        
        Args:
            query: Foydalanuvchi so'rovi
            
        Returns:
            bool: Agentga tegishli bo'lsa True, aks holda False
        """
        # Agentga tegishli so'rovlarni aniqlash uchun kalit so'zlar
        # SOATO tool uchun yangi kalit so'zlar qo'shildi
        agent_keywords = [
            "soato", "mhobit", "viloyat", "tuman", "shahar", "millat", "mst", "ckp", 
            "mahsulot", "tasnif", "dbibt", "tashkilot", "okpo", "ktut", "stir", "inn",
            "статистика", "статистик", "statistika", "statistik",
            "davlat", "mamlakat", "country", "kod", 
            "markaz", "markazi", "ma'muriy", "aholi punkti", "tumanlari", "tumanlar"
        ]
        
        # So'rovni kichik harflarga o'tkazib, kalit so'zlarni tekshirish
        query_lower = query.lower()
        for keyword in agent_keywords:
            if keyword.lower() in query_lower:
                return True
                
        return False
    
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
        role = message_dict.get("role", "")
        content = message_dict.get("content", "")
        
        if role == "system":
            return SystemMessage(content=content)
        elif role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        else:
            # Noma'lum turlar uchun default HumanMessage
            return HumanMessage(content=content)
    
    async def stream_chat(self, context: str, query: str, language: str = "uz") -> AsyncGenerator[str, None]:
        """
        Modelga savol yuborish va javobni stream qilish
        
        Args:
            prompt (str): Savol matni
        
        Yields:
            str: Model javobining qismlari
        """
        messages = self._create_messages(context, query, language)
        async for chunk in self._stream(messages):
            yield chunk
    
    async def chat(self, context: str, query: str, language: str = "uz"):
        """
        Modelga savol yuborish va javob olish
        
        Args:
            prompt (str): Savol matni
        
        Returns:
            str: Model javobi
        """
        # Agent ishlatish yoki yo'qligini tekshirish
        if self.use_agent and self.agent and self._is_agent_query(query):
            try:
                logger.info(f"So'rov agentga yo'naltirildi (session: {self.session_id})")
                result = self.agent.invoke({"input": query})
                return result["output"]
            except Exception as e:
                logger.error(f"Agent xatoligi: {str(e)}, modelga o'tilmoqda")
                # Agent xatolik bersa, modelga o'tish
        
        # Agentga tegishli bo'lmagan so'rovlar yoki agent xatolik bergan holda model ishlatiladi
        messages = self._create_messages(context, query, language)
        return await self._invoke(messages)
    
    def _create_messages(self, system_prompts: str, user_prompt: str, language: str = "uz", device: str = "web") -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        detect_lang = {
            "uz": "uzbek",
            "ru": "russian",
        }
        messages = []

        prompts = self.default_system_prompts.copy()
        if device == "web":
            prompts.extend(self.system_html_prompts)
        else:
            prompts.extend(self.system_markdown_prompts)

        # System promptlarni qo'shish
        messages.append(SystemMessage(content="\n".join(prompts).format(language=detect_lang[language])))
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

    async def chat_stream_old(self, context: str, query: str, language: str = "uz", device: str = "web") -> AsyncGenerator[str, None]:
        """
        Javobni SSE orqali stream ko'rinishida yuborish
        
        Args:
            context (str): System prompt
            query (str): User savoli
            language (str): Til
            device (str): Qurilma
            
        Yields:
            str: Modeldan kelayotgan har bir token yoki parcha
        """
        # Agent ishlatish yoki yo'qligini tekshirish
        if self.use_agent and self.agent and self._is_agent_query(query):
            try:
                logger.info(f"So'rov agentga yo'naltirildi (stream) (session: {self.session_id})")
                # Agent javobini to'liq olish
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.agent.invoke({"input": query})
                    ), 
                    timeout=60.0
                )
                
                # Javobni tokenlar bo'yicha ajratib stream qilish
                response_text = result.get("output", "") if isinstance(result, dict) else str(result)
                response_text = response_text.replace("\n", "<br>")
                
                # Javobni tokenlar bo'yicha stream qilish
                import time
                for char in response_text:
                    yield char
                    await asyncio.sleep(0.01)  # Kichik kechikish stream effekti uchun
                return
            except asyncio.TimeoutError:
                logger.error(f"Agent timeout (stream): {self.session_id}")
                yield "Agent javob berish vaqti tugadi. Modelga o'tilmoqda..."
            except Exception as e:
                logger.error(f"Agent xatoligi (stream): {str(e)}, modelga o'tilmoqda")
                # Agent xatolik bersa, modelga o'tish
        
        # Agentga tegishli bo'lmagan so'rovlar yoki agent xatolik bergan holda model ishlatiladi
        messages = self._create_messages(context, query, language, device)
        async for chunk in self._stream_invoke(messages):
            yield chunk
    
    async def chat_stream(
        self, context: str, query: str, language: str = "uz", device: str = "web"
    ) -> AsyncGenerator[str, None]:
        agent_output = ""
        
        # 1. Agentdan javob olish
        if self.use_agent and self.agent and self._is_agent_query(query):
            try:
                logger.info("Agent ishlamoqda...")
                agent_result = self.agent.invoke({"input": query})
                agent_output = agent_result.get("output", "").strip()

                # Agar agent foydali javob bermasa — bo'sh qoldiramiz
                if not self._is_satisfactory(agent_output):
                    logger.warning("Agent javobi qoniqarsiz, model yakka ishlaydi.")
                    agent_output = ""

            except Exception as e:
                logger.error(f"Agent xatosi: {e}")
                agent_output = ""

        # 2. Agent javobi asosida model promptini yangilash
        if agent_output:
            combined_query = f"Savol: {query}\n\nAgent natijasi: {agent_output}\n\nYaxshilangan, aniq javob bering:"
        else:
            combined_query = query

        # 3. Modelni ishga tushirish
        messages = self._create_messages(context, combined_query, language, device)
        async for chunk in self._stream_invoke(messages):
            yield chunk


    async def get_stream_suggestion_question(self, suggested_context: str, query: str, answer: str, language: str, device: str = "web") -> AsyncGenerator[str, None]:
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
            "Faqat **bitta** savolni tanlang. Yangi savol o‘ylab topmang, faqat taqdim etilgan savollar ichidan 1 tasini tanlang va o'zgartirmagan holatda taqdim eting.\n"
            "Sizga berilgan kontekstda mavjud bo‘lgan savollardan faqat 1 tasini tanlang. Faqat bitta savolni tanlang. Faqatgina savolning o'zini yozing, boshqa hech qanday ma'lumot qo'shib yozmang.\n\n"
            "Yangi Savol {language} tilida bo'lishi kerak.\n"
            "Agar foydalanuvchining so‘rovi noaniq bo‘lsa, uni aniqlashtirishni so‘rang.\n"
            "Agar foydalanuvchi savoli salomlashish yoki tanishish to'g'risida bo'lsa, statistika nizomi to'g'risidagi savolni yozing.\n"
            "Javobni italik harflar bilan yozing.\n"
        )

        if device == "web":
            system_prompt += "Javoblarni HTML formatida yozing. <i> </i> tegidan foydalanib. Har bir javob semantik va vizual ravishda aniq bo'lishi kerak."
        else:
            system_prompt += "Javoblarni Markdown formatida yozing.  _ kursiv matn uchun. Har bir javob semantik va vizual ravishda aniq bo'lishi kerak."

        if language == "ru":
            system_prompt += "\n\nPlease provide the response in Russian."

        full_prompt = (
            f"Taqdim etilgan savollar:\n{suggested_context}\n\n" 
            f"Foydalanuvchi savoli: {query}\n\n"
            f"Javob: {answer}\n\n"
            f"Yangi savol: "
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_prompt)
        ]
        if language == "ru":
            messages.append(HumanMessage(content="Answer translate the result into Russian for me."))

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

    def _is_satisfactory(self, response: str) -> bool:
        """
        Agent javobining qoniqarli ekanligini tekshirish
        
        Args:
            response (str): Agent javobi
            
        Returns:
            bool: Javob qoniqarli bo'lsa True, aks holda False
        """
        if not response or len(response) < 10:
            return False
            
        # Xatolik xabarlarini tekshirish
        error_indicators = [
            "xatolik", "error", "topilmadi", "not found", 
            "mavjud emas", "uzr", "sorry", "kechirasiz"
        ]
        
        response_lower = response.lower()
        for indicator in error_indicators:
            if indicator in response_lower:
                return False
                
        # Minimal uzunlik tekshiruvi (kamida 50 belgi)
        if len(response) < 50:
            return False
            
        return True

# Factory funksiya - model obyektini olish
@lru_cache(maxsize=10)  # Eng ko'p 10 ta sessiya uchun cache
def get_model_instance(session_id: Optional[str] = None, model_name: str = "devstral:latest", base_url: str = "http://localhost:11434", use_agent: bool = True) -> LangChainOllamaModel:
    return LangChainOllamaModel(session_id=session_id, model_name=model_name, base_url=base_url, use_agent=use_agent)

# Asosiy model obyekti (eski kod bilan moslik uchun)
model = get_model_instance()