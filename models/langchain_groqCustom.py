import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain importlari
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage, 
    SystemMessage, 
    BaseMessage
)
from langchain_groq import ChatGroq

# .env faylidan API kalitni o'qish
load_dotenv()

class LangChainGroqModel:
    """Groq bilan LangChain integratsiyasi"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Asosiy model nomi (boshqa modellar kommentlarda)
        self.model_name = "llama3-70b-8192"
        
        # LangChain ChatGroq modelini yaratish
        self.model = ChatGroq(
            model_name=self.model_name,
            groq_api_key=self.api_key,
            temperature=0.7,
            max_tokens=1000
        )
        
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
            "If the answer is not clear, add the sentence \"Please clarify the question\""
        ]
            
    def _get_message_type(self, message: BaseMessage) -> str:
        """Message turini aniqlash"""
        if isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return "unknown"
            
    def _create_message_from_dict(self, message_dict: Dict[str, Any]) -> BaseMessage:
        """Dictionary ma'lumotlaridan message obyekti yaratish"""
        msg_type = message_dict.get("type", "")
        content = message_dict.get("content", "")
        
        if msg_type == "human":
            return HumanMessage(content=content)
        elif msg_type == "ai":
            return AIMessage(content=content)
        elif msg_type == "system":
            return SystemMessage(content=content)
        else:
            # Noma'lum tur uchun default
            return HumanMessage(content=content)
    
    async def chat(self, context: str, query: str) -> str:
        """
        Modelga savol yuborish va javob olish

        Args:
            prompt (str): Savol matni

        Returns:
            str: Model javobi
        """
        
        # Kontekst tarixisiz xabarlarni tayyorlash
        messages = self._create_messages(
            context,
            query
        )
        
        # Modeldan javob olish
        one_result = await self._invoke(messages)
        
        return one_result
        
    def _create_messages(self, system_prompts: List[str], user_prompt: str) -> List[BaseMessage]:
        """
        System va user promptlaridan LangChain message obyektlarini yaratish
        """
        messages = []
        
        # System promptlarini qo'shish
        # for prompt in system_prompts:
        messages.append(SystemMessage(content="\n".join(self.default_system_prompts)))
        messages.append(SystemMessage(content=system_prompts))
        
        # User promptini qo'shish
        messages.append(HumanMessage(content=user_prompt))
        
        return messages
        
    async def _invoke(self, messages: List[BaseMessage]) -> str:
        """
        LangChain orqali modelni chaqirish
        """
        try:
            # LangChain model chaqiruvi
            result = await self.model.ainvoke(messages)
            return result.content
        except Exception as e:
            print("Error in invoke:", str(e))
            raise e
    
    async def logical_context(self, text: str) -> str:
        """
        Berilgan matnni mantiqiy qayta ishlash
        """
        try:
            system_prompts = [
                "Let the information be based primarily on context and relegate additional answers to a secondary level.",
                "Don't add unrelated context",
                f"From the following text: 1. Remove sentences that are not consistent in content (irrelevant or off-topic), 2. Remove repeated sentences. Don't mix up the sentences. Rewrite the text in a logically consistent and simplified way: TEXT:{text}",
                "Please provide the answer in Uzbek.",
                "Faqat o'zbek tilida javob ber.",
                "Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma",
                "Use the same meaning only once.",
                "Bir xil manodagi gaplar bo'lsa ularni qisqartir",
                "I will not answer you, the answer is as follows, without words, the answer is self-replying",
                "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"
            ]
            
            # Kontekst tarixisiz xabarlarni tayyorlash
            messages = self._create_messages(system_prompts, "")
            
            result = await self.model.ainvoke(messages)
            return result.content
        except Exception as e:
            print("Error in logical_context:", str(e))
            raise e
    
    async def generate_questions(self, question: str) -> list:
        """
        Berilgan savolga asoslanib 3 ta sinonim savol yaratish
        """
        prompt = f"""Quyidagi savolga 3 ta sinonim shaklida savol yozing. Har bir sinonim asl savol bilan bir xil ma'noni bildirishi kerak, lekin turlicha ifodalangan bo'lsin. Keraksiz yoki qo'shimcha ma'no qo'shmang.
                    Ma'lumotni list ko'rinishida ber
                    Salomlashish va shunga o'xshagan savollar bo'lsa bosh array qaytar
                    savollarni boshiga nmer qo'ymang
                Savol: {question}"""
        
        try:
            # Faqat user prompt beriladi, system prompt yo'q
            messages = [HumanMessage(content=prompt)]
            
            # Modelni chaqirish
            result = await self.model.ainvoke(messages)
            
            # Javobni qatorlarga bo'lish
            sentences = [s.strip() for s in result.content.split('\n') if s.strip()]
            return sentences[:3]  # Eng ko'pi bilan 3 ta natija qaytarish
        except Exception as e:
            print(f"Error in generating sentences: {str(e)}")
            return ["Error generating sentences.", "Please try again.", "Check your connection."]
    
    async def rewrite_query(self, user_query: str, chat_history: str) -> dict:
        try:
            # Agar tarix bo'sh bo'lsa yoki so'rov juda qisqa bo'lsa, original so'rovni qaytarish
            if not chat_history or chat_history.strip() == "":
                return {"content": user_query}

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

            # Modeldan javob olish
            result = await self.model.ainvoke([system_message, user_message])
            
            # Javobni tahlil qilish va tozalash
            response_text = result.content.strip()
            
            # Prefiks va suffikslarni olib tashlash
            prefixes_to_remove = ["QAYTA YOZILGAN SAVOL:", "QAYTA YOZILGAN SAVOL", "Qayta yozilgan savol:", "SAVOL:", "Savol:"]
            for prefix in prefixes_to_remove:
                if response_text.startswith(prefix):
                    response_text = response_text[len(prefix):].strip()
            
            # HTML, markdown formatlarini tozalash
            response_text = response_text.replace("```html", "").replace("```", "")
            
            # Agar natija savol emas, balki javob ko'rinishida bo'lsa, original so'rovni qaytarish
            javob_belgilari = ["<p>", "</p>", "<b>", "</b>", "<i>", "</i>", "javob:", "Javob:"]
            if any(marker in response_text for marker in javob_belgilari) or len(response_text.split()) > 30:
                return {"content": user_query}  # Javob emas, original so'rovni qaytarish

            # Agar qayta yozilgan savol juda qisqa bo'lsa, original savolni qaytarish
            if len(response_text.split()) < 3:
                return {"content": user_query}

            return {"content": response_text}

        except Exception as e:
            print("Error in rewrite_query:", str(e))
            # Xatolik yuz berganda xavfsiz yo'l - original savolni qaytarish
            return {"content": user_query}

    def count_tokens(self, text: str) -> int:
        """
        Matndagi tokenlar sonini taxminiy hisoblash. 
        Bu oddiy taxminiy usul - aniq bo'lmagan lekin tezkor hisoblanadi.
        
        Args:
            text (str): Token soni hisoblanadigan matn
            
        Returns:
            int: Taxminiy token soni
        """
        # Oddiy tokenlashtirish qoidalari
        # 1. O'rtacha 1 token = 4 belgi ingliz tilida
        # 2. Har bir so'z taxminan 1.3 token
        # 3. Har bir tinish belgisi alohida token
        
        if not text:
            return 0
            
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

# Model obyektini yaratish (eski kod bilan bir xil nom)
model = LangChainGroqModel()
