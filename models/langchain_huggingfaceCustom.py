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
from langchain_huggingface import HuggingFacePipeline

# Transformers importlari
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .env faylidan konfiguratsiyani o'qish
load_dotenv()

# Global model va tokenizer obyektlarini saqlash uchun cache
_MODEL_CACHE = {}
_TOKENIZER_CACHE = {}
_PIPELINE_CACHE = {}

# Global semaphore - bir vaqtda nechta so'rov bajarilishi mumkinligini cheklaydi
_MODEL_SEMAPHORE = asyncio.Semaphore(3)  # Bir vaqtda 3 ta so'rovga ruxsat

class LangChainHuggingFaceModel:
    """Hugging Face bilan LangChain integratsiyasi - ko'p foydalanuvchilar uchun optimallashtirilgan"""

    # Sinf darajasidagi o'zgaruvchilar
    _instances: ClassVar[Dict[str, 'LangChainHuggingFaceModel']] = {}

    def __init__(self,
                 session_id: Optional[str] = None,
                 model_name: str = "google/gemma-2b-it",  # Kichikroq model
                 device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.float16,
                 trust_remote_code: bool = True,
                 temperature: float = 0.7,
                 max_tokens: int = 512):  # Kamroq tokenlar
        """
        Hugging Face modelini LangChain bilan ishlatish uchun klass.

        Args:
            session_id: Foydalanuvchi sessiyasi ID si
            model_name: Hugging Face model nomi
            device_map: Modelni qaysi qurilmada ishlatish
            torch_dtype: Model uchun data type
            trust_remote_code: Remote code ga ishonish
            temperature: Generatsiya uchun temperature
            max_tokens: Maksimal token uzunligi
        """
        # CUDA xatoligini oldini olish uchun
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        self.session_id = session_id or "default"
        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Model va tokenizerini olish
        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()
        self.pipe = self._get_pipeline()

        # LangChain HuggingFacePipeline yaratish
        self.hf_pipeline = HuggingFacePipeline(pipeline=self.pipe)

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

    def _get_tokenizer(self):
        """Model uchun tokenizer olish (cache dan yoki yangi yaratish)"""
        global _TOKENIZER_CACHE
        if self.model_name not in _TOKENIZER_CACHE:
            logger.info(f"Tokenizer yaratilmoqda: {self.model_name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    token=os.getenv("HF_TOKEN")  # API token
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                _TOKENIZER_CACHE[self.model_name] = tokenizer
            except Exception as e:
                logger.error(f"Tokenizer yaratishda xato: {str(e)}")
                raise
        return _TOKENIZER_CACHE[self.model_name]

    def _get_model(self):
        """Model olish (cache dan yoki yangi yaratish)"""
        global _MODEL_CACHE
        if self.model_name not in _MODEL_CACHE:
            logger.info(f"Model yuklanmoqda: {self.model_name}")
            try:
                # CUDA xotira tozalash
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Model konfiguratsiyasi - minimal
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "trust_remote_code": self.trust_remote_code,
                    "token": os.getenv("HF_TOKEN")
                }
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                _MODEL_CACHE[self.model_name] = model
                
            except Exception as e:
                logger.error(f"Model yuklashda xato: {str(e)}")
                raise
        return _MODEL_CACHE[self.model_name]

    def _get_pipeline(self):
        """Pipeline olish (cache dan yoki yangi yaratish)"""
        global _PIPELINE_CACHE
        if self.model_name not in _PIPELINE_CACHE:
            logger.info(f"Pipeline yaratilmoqda: {self.model_name}")
            try:
                # Pipeline konfiguratsiyasi - minimal
                pipe_kwargs = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "do_sample": True,
                    "return_full_text": False
                }
                
                pipe = pipeline(
                    "text-generation",
                    **pipe_kwargs
                )
                _PIPELINE_CACHE[self.model_name] = pipe
            except Exception as e:
                logger.error(f"Pipeline yaratishda xato: {str(e)}")
                raise
        return _PIPELINE_CACHE[self.model_name]

    def _get_message_type(self, message: BaseMessage) -> str:
        """Message turini aniqlash"""
        if isinstance(message, SystemMessage):
            return "system"
        elif isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
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
        return HumanMessage(content=content)

    async def chat(self, context: str, prompt: str) -> str:
        """Modelga savol yuborish va javob olish"""
        try:
            # Minimal kontekst va savol
            max_context_length = 512  # Kamroq kontekst
            context = context[:max_context_length]
            prompt = prompt[:256]  # Kamroq savol
            
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}"
            # Faqat eng muhim promptlar
            important_prompts = [
                self.default_system_prompts[0],  # Context bo'yicha javob
                self.default_system_prompts[6],  # O'zbek/rus tili
            ]
            messages = self._create_messages(important_prompts, full_prompt)
            
            response = await self._ainvoke(messages)
            return response
            
        except Exception as e:
            logger.error(f"Chat xatolik: {str(e)}")
            return f"Xatolik yuz berdi: {str(e)}"

    def _create_messages(self, system_prompts: List[str], user_prompt: str) -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        messages = []
        for prompt in system_prompts:
            messages.append(SystemMessage(content=prompt))
        messages.append(HumanMessage(content=user_prompt))
        return messages

    async def _ainvoke(self, messages: List[BaseMessage]) -> str:
        """
        LangChain orqali modelni chaqirish (semaphore bilan cheklangan)
        """
        formatted_prompt = self._format_messages_for_model(messages)
        async with _MODEL_SEMAPHORE:
            logger.info(f"Model chaqirilmoqda (session: {self.session_id})")
            try:
                outputs = await asyncio.to_thread(
                    self.pipe,
                    formatted_prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                response = outputs[0]["generated_text"].strip()
                return response
            except Exception as e:
                logger.error(f"Model chaqirishda xatolik: {str(e)}")
                return f"Xatolik yuz berdi: {str(e)}"

    def _format_messages_for_model(self, messages: List[BaseMessage]) -> str:
        """Xabarlarni model uchun formatlash"""
        formatted_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_parts.append(f"<|SYSTEM|> {message.content}")
            elif isinstance(message, HumanMessage):
                formatted_parts.append(f"<|USER|> {message.content}")
            elif isinstance(message, AIMessage):
                formatted_parts.append(f"<|ASSISTANT|> {message.content}")
        return "\n".join(formatted_parts)

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
        return await self._ainvoke(messages)

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
        response = await self._ainvoke(messages)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:3]

    async def rewrite_query(self, user_query: str, chat_history: str = "") -> Dict[str, str]:
        """
        Foydalanuvchi so'rovini qayta yozish

        Args:
            user_query: Foydalanuvchi so'rovi
            chat_history: Chat tarixi (ixtiyoriy)

        Returns:
            Dict[str, str]: Qayta yozilgan so'rov
        """
        try:
            system_message = SystemMessage(
                content=(
                    "ATTENTION! YOU ARE THE QUESTION REWRITER. DO NOT ANSWER, JUST REWRITE THE QUESTION!\n\n"
                    "Task: Rewrite the user's question in a MORE COMPLETE and SPECIFIC form using the context of the conversation.\n"
                    "Rules:\n"
                    "1. IMPORTANT: JUST REWRITE THE QUESTION. DO NOT ANSWER THE QUESTION!\n"
                    "2. ONLY ONE QUESTION RETURN. DO NOT ANSWER, EXPLANATION, TUTORIAL - ONLY QUESTION.\n"
                    "3. Indexes (bu, shu, u, ular, ushbu, o'sha) should be replaced with exact information.\n"
                    "4. If the previous conversation refers to a topic, specify it exactly.\n"
                    "5. Save the grammar structure of the question, only replace the index/unclear parts.\n\n"
                    "6. Answer in Uzbek.\n\n"
                    f"CHAT HISTORY:\n{chat_history}\n"
                )
            )
            user_message = HumanMessage(
                content=f"QAYTA YOZILADIGAN SAVOL: {user_query}\n\nQAYTA YOZILGAN SAVOL FORMAT: faqat savol berilishi kerak, javob emas"
            )
            response_text = await self._ainvoke([system_message, user_message])
            response_text = response_text.strip()
            prefixes_to_remove = ["QAYTA YOZILGAN SAVOL:", "QAYTA YOZILGAN SAVOL", "Qayta yozilgan savol:", "SAVOL:", "Savol:"]
            for prefix in prefixes_to_remove:
                if response_text.startswith(prefix):
                    response_text = response_text[len(prefix):].strip()
            response_text = response_text.replace("```html", "").replace("```", "")
            javob_belgilari = ["<p>", "</p>", "<b>", "</b>", "<i>", "</i>", "javob:", "Javob:"]
            if any(marker in response_text for marker in javob_belgilari) or len(response_text.split()) > 30:
                return {"content": user_query}
            if len(response_text.split()) < 3:
                return {"content": user_query}
            return {"content": response_text}
        except Exception as e:
            logger.error(f"Error in rewrite_query: {str(e)}")
            return {"content": user_query}

    def count_tokens(self, text: str) -> int:
        """
        Matndagi tokenlar sonini hisoblash.
        Avval tokenizer orqali aniq hisoblash, agar ishlamasa taxminiy hisoblash.

        Args:
            text (str): Token soni hisoblanadigan matn

        Returns:
            int: Token soni
        """
        if not text:
            return 0
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Tokenizer xatoligi: {str(e)}, taxminiy hisoblash ishlatiladi")
            words = len(text.split())
            punctuation = sum(1 for char in text if char in '.,;:!?()-"\'[]{}')
            chars = len(text)
            estimate1 = chars / 4
            estimate2 = words * 1.3 + punctuation
            result = int((estimate1 + estimate2) / 2)
            return max(1, result)


@lru_cache(maxsize=10)
def get_model_instance(session_id: Optional[str] = None, model_name: str = "google/gemma-2b-it"):
    """
    Model obyektini olish (mavjud bo'lsa cache dan, aks holda yangi yaratish)

    Args:
        session_id: Foydalanuvchi sessiyasi ID si
        model_name: Hugging Face model nomi

    Returns:
        LangChainHuggingFaceModel: Model obyekti
    """
    logger.info(f"Model obyekti so'raldi: {session_id}")
    return LangChainHuggingFaceModel(session_id=session_id, model_name=model_name)


model = get_model_instance()