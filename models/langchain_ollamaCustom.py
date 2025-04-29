import os
import json
import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, ClassVar
from dotenv import load_dotenv
from functools import lru_cache
import torch

# LangChain importlari
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_models import ChatOllama

# .env faylidan konfiguratsiyani o'qish
load_dotenv()

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model va pipeline obyektlarini saqlash uchun cache
_MODEL_CACHE: Dict[str, ChatOllama] = {}

# Global semaphore - bir vaqtda nechta so'rov bajarilishi mumkinligini cheklaydi
_MODEL_SEMAPHORE = asyncio.Semaphore(10)  # Bir vaqtda 10 ta so'rovga ruxsat

class LangChainOllamaModel:
    """Ollama bilan LangChain integratsiyasi - ko'p foydalanuvchilar uchun optimallashtirilgan"""
    
    # Sinf darajasidagi o'zgaruvchilar
    _instances: ClassVar[Dict[str, 'LangChainOllamaModel']] = {}
    
    def __init__(self, 
                 session_id: Optional[str] = None,
                 model_name: str = "gemma3:27b",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 num_ctx: int = 8192,          # Kontekst oynasini maksimal qilish
                 num_gpu: int = 1,             # Barcha mavjud GPU larni ishlatish
                 num_thread: int = 16,         # CPU thread larni ko'paytirish
                 num_batch: int = 8,           # Batch size ni oshirish
                 f16: bool = True,             # FP16 tezlashtirish
                 rope_frequency_base: float = 1e6,  # RoPE chastotasini optimallashtirish
                 rope_frequency_scale: float = 1.0):
        """
        Ollama modelini LangChain bilan ishlatish uchun klass.
        
        Args:
            session_id: Sessiya identifikatori
            model_name: Ollama model nomi
            base_url: Ollama server URL
            temperature: Generatsiya temperaturasi
            num_ctx: Kontekst oynasi o'lchami
            num_gpu: GPU lar soni
            num_thread: CPU thread lar soni
            num_batch: Batch size
            f16: FP16 tezlashtirish
            rope_frequency_base: RoPE chastotasi
            rope_frequency_scale: RoPE masshtabi
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self.num_thread = num_thread
        self.num_batch = num_batch
        self.f16 = f16
        self.rope_frequency_base = rope_frequency_base
        self.rope_frequency_scale = rope_frequency_scale
        
        # Model yaratish yoki cache dan olish
        self.model = self._get_model()
        
        # System prompts ro'yxati (soddalashtirilgan)
        self.default_system_prompts = [
            "Faqat berilgan kontekst asosida javob bering.",
            "Javoblar O'zbekiston Respublikasi Milliy Statistik Qo'mitasi ma'lumotlari doirasida bo'lishi kerak.",
            "Javobni faqat o'zbek tilida, xatosiz yozing.",
            "Javobda <p>, <ul>, <li>, <strong> kabi HTML teglaridan foydalaning.",
            "Agar savol aniq bo'lmasa, 'Iltimos, savolni aniqlashtiring' deb javob bering."
        ]
    
    def _get_model(self) -> BaseChatModel:
        """Model olish (cache dan yoki yangi yaratish)"""
        global _MODEL_CACHE
        
        cache_key = f"{self.model_name}_{self.session_id}"
        
        if cache_key not in _MODEL_CACHE:
            logger.info(f"Ollama model yaratilmoqda: {self.model_name}")
            try:
                # CUDA va GPU ni tekshirish
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU topilmadi! CUDA o'rnatilganligini tekshiring.")
                
                # GPU xotirasini tozalash
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # GPU device ni olish
                device = torch.device("cuda:0")
                torch.cuda.set_device(device)
                
                # Model konfiguratsiyasi
                model_config = {
                    "model": self.model_name,
                    "base_url": self.base_url,
                    "temperature": self.temperature,
                    "num_ctx": 8192,
                    "num_gpu": 1,
                    "num_thread": 8,         # CPU thread larni kamaytirish
                    "num_batch": 512,
                    "f16": True,
                    "extra_model_kwargs": {
                        "gpu_layers": -1,
                        "gpu_memory": 25000,  # 25GB VRAM
                        "cpu_memory": 0,  # CPU xotirasini o'chirish
                        "compute_type": "float16",
                        "use_cuda": True,
                        "use_gpu": True,
                        "max_parallel": 64,
                        "batch_size": 512,
                        "max_tokens": 8192,
                        "max_gpu_memory": "25GiB",  # 25GB VRAM
                        "max_cpu_memory": "0GiB",  # CPU xotirasini o'chirish
                        "mmap": False,
                        "mlock": True,
                        "numa": False,
                        "threads": 8,
                        "tensor_split": [100],  # 100% GPU
                        "gpu_split_auto": False,
                        "n_gpu_layers": -1,
                        "offload_kqv": False,  # CPU ga o'tkazishni o'chirish
                        "no_mmap": True,
                        "n_ctx": 8192,
                        "rope_scaling": {"type": "linear", "factor": 4.0},
                        "pre_load": True,
                        "gpu_init": True,
                        "load_in_8bit": False,
                        "seed": 42,
                        "numa_strategy": "NUMA_STRATEGY_DISABLED",
                        "gpu_memory_utilization": 0.95,  # 95% VRAM ishlatish
                        "no_offload": True,  # CPU ga o'tkazishni o'chirish
                        "max_batch_total_tokens": 65536,
                    }
                }
                
                # Modelni yaratish
                model = ChatOllama(**model_config)
                
                # Modelni GPU ga yuklash
                logger.info("Model GPU ga yuklanmoqda...")
                response = model.invoke("Test message to load model")
                logger.info("Model GPU ga yuklandi")
                
                _MODEL_CACHE[cache_key] = model
                
                # GPU xotirasi monitoringi
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
                
                logger.info(f"GPU xotirasi:")
                logger.info(f"- Allocated: {allocated:.2f}GB")
                logger.info(f"- Reserved: {reserved:.2f}GB")
                logger.info(f"- Max used: {max_memory:.2f}GB")
                
            except Exception as e:
                logger.error(f"Model yaratishda xatolik: {str(e)}")
                if "404" in str(e):
                    logger.error(f"Model '{self.model_name}' topilmadi. Modelni yuklash kerak:")
                    logger.error(f"ollama pull {self.model_name}")
                raise
        
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
            return HumanMessage(content=content)
    
    async def chat(self, context: str, query: str) -> str:
        """
        Modelga savol yuborish va javob olish
        
        Args:
            context: Kontekst matni
            query: Foydalanuvchi savoli
        
        Returns:
            str: Model javobi (HTML formatida)
        """
        # Kontekst va savol uzunligini tekshirish
        total_tokens = self.count_tokens(context + query)
        if total_tokens > self.num_ctx:
            logger.warning(f"Kontekst uzunligi chegaradan oshdi: {total_tokens}/{self.num_ctx}")
            return "<p><strong>Xatolik:</strong> Kontekst yoki savol juda uzun. Iltimos, qisqartiring.</p>"
        
        # System va user promptlaridan xabarlar yaratish
        messages = self._create_messages(context, query)
        
        # Modelni chaqirish
        response = await self._invoke(messages)
        
        # Javobni HTML formatiga keltirish
        formatted_response = self._format_response(response)
        return formatted_response
    
    def _create_messages(self, context: str, query: str) -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        messages = []
        
        # System promptlarni qo'shish
        system_prompt = "\n".join([
            "Quyidagi kontekst asosida savolga javob bering:",
            "Kontekst bo'yicha aniq va to'g'ri javob bering.",
            "Agar kontekstda ma'lumot bo'lmasa, shuni aytib o'ting.",
            "Javobni HTML formatida qaytaring."
        ])
        messages.append(SystemMessage(content=system_prompt))
        
        # Kontekstni qo'shish
        if context.strip():
            context_prompt = f"KONTEKST:\n{context}\n\nSAVOL:\n{query}"
            messages.append(HumanMessage(content=context_prompt))
        else:
            messages.append(HumanMessage(content=query))
        
        return messages
    
    async def _invoke(self, messages: List[BaseMessage]) -> str:
        """
        LangChain orqali modelni chaqirish (semaphore bilan cheklangan)
        """
        async with _MODEL_SEMAPHORE:
            logger.info(f"Model chaqirilmoqda (session: {self.session_id})")
            
            try:
                response = await self.model.ainvoke(messages)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.error(f"Model chaqirishda xatolik: {str(e)}")
                return f"<p><strong>Xatolik:</strong> Javob olishda xatolik yuz berdi: {str(e)}</p>"
    
    def _format_response(self, response: str) -> str:
        """
        Javobni HTML formatiga keltirish
        """
        if not response.strip():
            return "<p>Iltimos, savolni aniqlashtiring.</p>"
        
        # Oddiy formatlash: har bir qatorni <p> tegi bilan o'rash
        lines = response.split('\n')
        formatted_lines = [f"<p>{line}</p>" for line in lines if line.strip()]
        
        # Agar ro'yxat bo'lsa, <ul><li> ga aylantirish
        if any(line.strip().startswith('-') for line in lines):
            formatted_lines = ["<ul>"]
            for line in lines:
                if line.strip().startswith('-'):
                    formatted_lines.append(f"<li>{line.strip()[1:].strip()}</li>")
                else:
                    formatted_lines.append(f"<p>{line}</p>")
            formatted_lines.append("</ul>")
        
        return "".join(formatted_lines)
    
    async def logical_context(self, text: str) -> str:
        """
        Berilgan matnni mantiqiy qayta ishlash
        """
        system_prompt = """
        Faqat berilgan matn asosida mantiqiy va faktlarga asoslangan ma'lumotlarni qayta ishlang.
        Ma'lumotlarni aniq, qisqa va HTML formatida (<p>, <ul>, <li>) taqdim eting.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text)
        ]
        
        response = await self._invoke(messages)
        return self._format_response(response)
    
    async def generate_questions(self, question: str) -> List[str]:
        """
        Berilgan savolga asoslanib 3 ta sinonim savol yaratish
        """
        system_prompt = """
        Berilgan savolga o'xshash, lekin boshqacha so'z birikmalaridan foydalangan holda 3 ta savol yarating.
        Savollar original savol bilan bir xil ma'noni anglatishi kerak.
        Har bir savolni alohida qatorda qaytaring.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original savol: {question}")
        ]
        
        response = await self._invoke(messages)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:3]
    
    async def rewrite_query(self, user_query: str, chat_history: str = "") -> str:
        """
        Foydalanuvchi so'rovini qayta yozish
        """
        system_prompt = """
        Chat tarixini hisobga olib, foydalanuvchi so'rovini kontekstga mos ravishda qayta yozing.
        Qayta yozilgan so'rov aniq va qisqa bo'lishi kerak.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chat tarixi:\n{chat_history}\n\nAsl so'rov: {user_query}")
        ]
        
        try:
            response = await self._invoke(messages)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            logger.error(f"So'rovni qayta yozishda xatolik: {str(e)}")
            return user_query
    
    def count_tokens(self, text: str) -> int:
        """
        Matndagi tokenlar sonini taxminiy hisoblash
        """
        if not text:
            return 0
        
        words = len(text.split())
        punctuation = sum(1 for char in text if char in '.,;:!?()-"\'[]{}')
        chars = len(text)
        
        estimate1 = chars / 4
        estimate2 = words * 1.3 + punctuation
        result = int((estimate1 + estimate2) / 2)
        
        return max(1, result)

# Factory funksiya - model obyektini olish
def get_model_instance(
    session_id: Optional[str] = None,
    model_name: str = "gemma3:27b",
    base_url: str = "http://localhost:11434"
) -> 'LangChainOllamaModel':
    """
    Model obyektini olish (mavjud bo'lsa cache dan, aks holda yangi yaratish)
    
    Args:
        session_id: Foydalanuvchi sessiyasi ID si
        model_name: Ollama model nomi
        base_url: Ollama server URL
    
    Returns:
        LangChainOllamaModel: Model obyekti
    """
    cache_key = f"{model_name}_{session_id or 'default'}"
    
    if cache_key not in LangChainOllamaModel._instances:
        logger.info(f"Yangi model yaratilmoqda: {model_name}")
        model = LangChainOllamaModel(
            session_id=session_id,
            model_name=model_name,
            base_url=base_url
        )
        LangChainOllamaModel._instances[cache_key] = model
    
    return LangChainOllamaModel._instances[cache_key]

# Asosiy model obyekti
model = get_model_instance()

# Test funksiyasi (ixtiyoriy)
if __name__ == "__main__":
    async def test_chat():
        ollama_model = get_model_instance()
        context = "O'zbekiston aholisi 2023 yilda 36 million kishi edi."
        query = "O'zbekiston aholisi haqida ma'lumot bering."
        response = await ollama_model.chat(context, query)
        print(response)
    
    asyncio.run(test_chat())