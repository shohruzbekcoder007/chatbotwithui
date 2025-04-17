"""
LangChain-compatible HuggingFace integration for Gemma models.
This module integrates the HuggingFace models with LangChain framework.
"""

import os
import asyncio
from typing import List, Optional, Any, Dict, Union

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_huggingface import HuggingFacePipeline

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer
)

class LangChainGemmaModel(BaseChatModel):
    """LangChain-compatible HuggingFace Gemma model."""
    
    def __init__(
        self, 
        model_name: str = "google/gemma-2b", 
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
        force_download: bool = True,
        temperature: float = 0.7,
        max_length: int = 512,
    ):
        """Initialize the LangChain HuggingFace model.
        
        Args:
            model_name (str): HuggingFace model identifier
            device_map (str): Device mapping strategy for model loading
            torch_dtype (torch.dtype): Data type for model weights
            trust_remote_code (bool): Whether to trust remote code
            force_download (bool): Whether to force download model files
            temperature (float): Temperature for generation
            max_length (int): Maximum length for generation
        """
        super().__init__()
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=trust_remote_code,
            force_download=force_download
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            force_download=force_download
        )
        
        # Create text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Create LangChain HuggingFacePipeline
        self.hf_pipeline = HuggingFacePipeline(pipeline=self.pipe)
    
    def _create_prompt_from_messages(self, messages: List[Union[SystemMessage, HumanMessage]]) -> str:
        """Create a formatted prompt from a list of messages.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        context = ""
        user_query = ""
        
        for message in messages:
            if isinstance(message, SystemMessage):
                context += message.content + "\n"
            elif isinstance(message, HumanMessage):
                user_query = message.content
        
        if context:
            prompt_parts.append(f"Ma'lumot:\n{context}")
        
        prompt_parts.append(f"Savol:\n{user_query}")
        prompt_parts.append("Javob:")
        
        return "\n\n".join(prompt_parts)
    
    def _create_messages(self, system_prompt: str, user_prompt: str) -> List[Union[SystemMessage, HumanMessage]]:
        """Create a list of LangChain message objects.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            
        Returns:
            List of LangChain message objects
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))
        return messages
    
    def _invoke(self, messages: List[Union[SystemMessage, HumanMessage]]) -> str:
        """Invoke the model with the given messages.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            Generated text response
        """
        prompt = self._create_prompt_from_messages(messages)
        response = self.hf_pipeline.invoke(prompt)
        # Strip the original prompt from the response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
        
    async def _ainvoke(self, messages: List[Union[SystemMessage, HumanMessage]]) -> str:
        """Asynchronously invoke the model with the given messages.
        
        Args:
            messages: List of LangChain message objects
            
        Returns:
            Generated text response
        """
        # Use asyncio to run the synchronous invoke in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._invoke, messages)
        return response
    
    def _generate(
        self, 
        messages: List[Union[SystemMessage, HumanMessage]], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None, 
        **kwargs: Any
    ) -> ChatResult:
        """Generate a response from the model.
        
        Args:
            messages: List of LangChain message objects
            stop: List of stop sequences
            run_manager: Run manager for callbacks
            kwargs: Additional keyword arguments
            
        Returns:
            ChatResult object with the generated response
        """
        text = self._invoke(messages)
        
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self, 
        messages: List[Union[SystemMessage, HumanMessage]], 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[Any] = None, 
        **kwargs: Any
    ) -> ChatResult:
        """Asynchronously generate a response from the model.
        
        Args:
            messages: List of LangChain message objects
            stop: List of stop sequences
            run_manager: Run manager for callbacks
            kwargs: Additional keyword arguments
            
        Returns:
            ChatResult object with the generated response
        """
        text = await self._ainvoke(messages)
        
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    async def chat(self, prompt: str, context: str = "") -> str:
        """Generate a response for the given prompt and context.
        
        Args:
            prompt: User prompt
            context: Additional context or system prompt
            
        Returns:
            Generated response text
        """
        messages = self._create_messages(context, prompt)
        return await self._ainvoke(messages)
    
    async def rewrite_query(self, query: str) -> str:
        """Rewrite the query for better retrieval.
        
        Args:
            query: Original query string
            
        Returns:
            Rewritten query
        """
        system_prompt = """
        Sen so'rovlarni qayta yozish asistentisan. Foydalanuvchilar so'rovlariga ko'ra ma'lumotlar bazasidan qidirishni yaxshilash uchun
        so'rovlarni kengaytirib yozib ber. So'rovni qayta yozganda:
        1. Asosiy ma'noni saqlash kerak
        2. Sinonimlardan foydalanish mumkin
        3. So'rovdagi muhim terminlarni kengaytirish kerak
        4. So'rovni yanada aniqlashtirish mumkin
        5. Faqat qayta yozilgan so'rovni qaytarish, boshqa izoh bermang
        """
        messages = self._create_messages(system_prompt, query)
        return await self._ainvoke(messages)
    
    async def logical_context(self, topic: str) -> str:
        """Generate logical context for a given topic.
        
        Args:
            topic: Topic to generate context about
            
        Returns:
            Generated context
        """
        system_prompt = """
        Sen matnga oid kontekstni yaratuvchi asistentsan. Berilgan mavzu asosida mantiqiy va faktlarga asoslangan 
        ma'lumotlarni tuzib berishing kerak. Faqat haqiqiy faktlarni ishlatib, barcha ma'lumotlarni aniq va qisqa shaklda yozib ber.
        """
        messages = self._create_messages(system_prompt, f"Quyidagi mavzu haqida mantiqiy kontekst yaratib ber: {topic}")
        return await self._ainvoke(messages)
    
    async def generate_questions(self, topic: str, num_questions: int = 3) -> List[str]:
        """Generate questions about a topic.
        
        Args:
            topic: Topic to generate questions about
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        system_prompt = f"""
        Sen savol generatsiya qiluvchi asistentsan. Berilgan mavzu asosida {num_questions} ta savolni generatsiya qil.
        Savollar mavzuga tegishli, qiziqarli va o'ylashga undovchi bo'lishi kerak. Faqat savollarning o'zini qaytaring,
        har bir savolni alohida qatorda.
        """
        messages = self._create_messages(system_prompt, f"Mavzu: {topic}")
        response = await self._ainvoke(messages)
        
        # Split by newlines and filter out empty lines
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        
        # Limit to num_questions
        return questions[:num_questions]
    
    def generate_text(self, prompt: str, context: str = "", max_length: int = None, temperature: float = None) -> str:
        """Generate text for a given prompt and context.
        
        Args:
            prompt: User prompt
            context: Additional context
            max_length: Maximum token length for generation (overrides instance default)
            temperature: Temperature for generation (overrides instance default)
            
        Returns:
            Generated text response
        """
        # Create formatted prompt
        full_prompt = ""
        if context:
            full_prompt += f"Ma'lumot:\n{context}\n\n"
        full_prompt += f"Savol:\n{prompt}\n\nJavob:"
        
        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_length or self.max_length,
            "temperature": temperature or self.temperature,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Generate text
        outputs = self.pipe(
            full_prompt,
            **gen_kwargs
        )
        
        # Extract the generated text
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt from the response
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt):].strip()
            
        return generated_text

# Create model instance
model = LangChainGemmaModel()
