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
                model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                tensor_parallel_size: int = 1,
                gpu_memory_utilization: float = 0.9,
                max_model_len: int = 4096,
                trust_remote_code: bool = True,
                temperature: float = 0.7,
                max_tokens: int = 1000):
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
        """
        self.session_id = session_id or "default"
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Model yaratish yoki cache dan olish
        self.model = self._get_model()
        
        # System prompts ro'yxati
        self.default_system_prompts = [
            "Let the information be based primarily on context and relegate additional answers to a secondary level.",
            "Do not include unrelated context and present it as separate information.",
            "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI.",
            "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan.",
            "You must respond only in Uzbek. Ensure there are no spelling mistakes.",
            "You should generate responses strictly based on the given prompt information without creating new content on your own.",
            "You are only allowed to answer questions related to the National Statistics Committee.",
            "The questions are within the scope of the estate and reports.",
            "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)",
            "Faqat o'zbek tilida javob ber.",
            "Integrate information that is not available in context. Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma",
            "Don't make up your own questions and answers, just use the information provided. O'zing savolni javobni to'qib chiqarma faqat berilgan ma'lumotlardan foydalan."
        ]
    
    def _get_model(self):
        """Model olish (cache dan yoki yangi yaratish)"""
        global _MODEL_CACHE
        
        cache_key = f"{self.model_name}_{self.tensor_parallel_size}"
        
        if cache_key not in _MODEL_CACHE:
            logger.info(f"VLLM model yaratilmoqda: {self.model_name}")
            model = VLLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=self.trust_remote_code,
                temperature=self.temperature,
                max_tokens=self.max_tokens
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
    
    async def chat(self, prompt: str) -> str:
        """
        Modelga savol yuborish va javob olish
        
        Args:
            prompt (str): Savol matni
        
        Returns:
            str: Model javobi
        """
        # System va user promptlaridan xabarlar yaratish
        messages = self._create_messages(self.default_system_prompts, prompt)
        
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