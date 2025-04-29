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
from langchain_community.llms import VLLM

# .env faylidan konfiguratsiyani o'qish
load_dotenv()

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model va pipeline obyektlarini saqlash uchun cache
_MODEL_CACHE = {}

# Global semaphore - bir vaqtda nechta so'rov bajarilishi mumkinligini cheklaydi
_MODEL_SEMAPHORE = asyncio.Semaphore(5)  # Bir vaqtda 5 ta so'rovga ruxsat

class LangChainVLLMModel:
    """VLLM bilan LangChain integratsiyasi - ko'p foydalanuvchilar uchun optimallashtirilgan"""
    
    # Sinf darajasidagi o'zgaruvchilar
    _instances: ClassVar[Dict[str, 'LangChainVLLMModel']] = {}
    
    def __init__(self, 
                session_id: Optional[str] = None,
                model_name: str = "google/gemma-3-27b-it",
                tensor_parallel_size: int = 1,
                gpu_memory_utilization: float = 0.75,  # 75% GPU xotirasi
                max_model_len: int = 4096,
                trust_remote_code: bool = True,
                temperature: float = 0.7,
                max_tokens: int = 1000,
                quantization: str = "awq",  # AWQ kvantlash
                block_size: int = 16):  # Block size
        """
        VLLM modelini LangChain bilan ishlatish uchun klass.
        
        Args:
            session_id: Foydalanuvchi sessiyasi ID si
            model_name: VLLM model nomi
            tensor_parallel_size: Tensor parallelism darajasi (GPU soni)
            gpu_memory_utilization: GPU xotira ishlatish foizi (0.0-1.0)
            max_model_len: Maksimal model uzunligi
            trust_remote_code: Remote code ga ishonish
            temperature: Generatsiya uchun temperature
            max_tokens: Maksimal token uzunligi
            quantization: Kvantlash turi ('awq', 'squeezellm', None)
            block_size: KV cache block size
        """
        self.session_id = session_id or "default"
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.quantization = quantization
        self.block_size = block_size
        
        # Model yaratish yoki cache dan olish
        self.model = self._get_model()
        
        # System prompts ro'yxati
        self.default_system_prompts = [
            "Let the information be based primarily on context and relegate additional answers to a secondary level.",
            "You are an AI assistant that helps users with their questions.",
            "You should generate responses strictly based on the given prompt information without creating new content on your own.",
            "You are only allowed to answer questions related to the National Statistics Committee.",
            "The questions are within the scope of the estate and reports.",
            "Output the results only in HTML format, no markdown, no latex. (only use these tags: <b></b>, <i></i>, <p></p>)",
            "O'zbek va rus tillarida javob ber.",
            "Отвечай на русском языке, если вопрос задан на русском.",
            "Integrate information that is not available in context.",
            "Don't make up your own questions and answers, just use the information provided."
        ]
    
    def _get_model(self):
        """Model olish (cache dan yoki yangi yaratish)"""
        global _MODEL_CACHE
        
        cache_key = f"{self.model_name}_{self.tensor_parallel_size}_{self.quantization}"
        
        if cache_key not in _MODEL_CACHE:
            logger.info(f"VLLM model yaratilmoqda: {self.model_name}")
            
            try:
                # Xotira monitoringi
                import psutil
                import torch
                
                # GPU xotira monitoringi
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    suggested_vram = int(gpu_memory * self.gpu_memory_utilization)
                    
                # VLLM model konfiguratsiyasi
                model_kwargs = {
                    "model": self.model_name,
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_model_len": self.max_model_len,
                    "trust_remote_code": self.trust_remote_code,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "quantization": self.quantization,
                    "block_size": self.block_size,
                    "dtype": "float16",  # FP16 tezroq va kamroq xotira
                    "max_num_batched_tokens": 4096,  # Batch size cheklovi
                    "max_num_seqs": 256,  # Maksimal parallel sequences
                    "disable_cache": False,  # KV-cache yoqilgan
                    "swap_space": 4,  # CPU RAM ga swap qilish (GB)
                    "max_num_cpu_threads": psutil.cpu_count(logical=False)  # Fizik CPU yadrolar
                }
                
                # Modelni yaratish
                model = VLLM(**model_kwargs)
                
                # Cache ga saqlash
                _MODEL_CACHE[cache_key] = model
                
                # Xotira holatini tekshirish
                if torch.cuda.is_available():
                    gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    if gpu_usage > 0.9:
                        logger.warning(f"Yuqori GPU xotira sarfi: {gpu_usage:.1%}")
                
                ram_usage = psutil.virtual_memory().percent
                if ram_usage > 90:
                    logger.warning(f"Yuqori RAM sarfi: {ram_usage}%")
                
            except Exception as e:
                logger.error(f"Model yaratishda xato: {str(e)}")
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
            # Noma'lum turlar uchun default HumanMessage
            return HumanMessage(content=content)
    
    async def chat(self, context: str, prompt: str) -> str:
        """
        Modelga savol yuborish va javob olish
        
        Args:
            context (str): Kontekst ma'lumotlari
            prompt (str): Savol matni
        
        Returns:
            str: Model javobi
        """
        # Kontekst va savolni birlashtirish
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
        
        # System va user promptlaridan xabarlar yaratish
        messages = self._create_messages(self.default_system_prompts, full_prompt)
        
        # Modelni chaqirish
        response = await self._invoke(messages)
        
        return response
    
    def _create_messages(self, system_prompts: List[str], user_prompt: str) -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        messages = []
        
        # System promptlarni qo'shish
        for prompt in system_prompts:
            messages.append(SystemMessage(content=prompt))
        
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
                # Xabarlarni model uchun formatlash
                formatted_prompt = self._format_messages_for_model(messages)
                
                # VLLM modelini chaqirish
                response = await self.model.ainvoke(formatted_prompt)
                return response
            except Exception as e:
                logger.error(f"Model chaqirishda xatolik: {str(e)}")
                return f"Xatolik yuz berdi: {str(e)}"
    
    def _format_messages_for_model(self, messages: List[BaseMessage]) -> str:
        """Xabarlarni model uchun formatlash"""
        formatted_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append(f"<|system|>\n{message.content}</s>")
            elif isinstance(message, HumanMessage):
                formatted_messages.append(f"<|human|>\n{message.content}</s>")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"<|ai|>\n{message.content}</s>")
        
        formatted_messages.append("<|ai|>")
        return "\n".join(formatted_messages)


model = LangChainVLLMModel(
    session_id="user123",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.8,
    max_model_len=4096,
    temperature=0.7,
    max_tokens=500,
    quantization="awq",
    block_size=16
)