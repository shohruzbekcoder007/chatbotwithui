import asyncio
from typing import List, Dict, Optional
from collections import defaultdict

# LangChain importlari
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langchain_ollama import ChatOllama


class LangChainOllamaModel:
    """Ollama bilan LangChain integratsiyasi (session kontekst bilan)"""

    def __init__(self):
        self.base_url = 'http://localhost:11434'
        self.model_name = 'gemma3:27b'

        # Ollama model
        self.model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
        )

        # System promptlar
        self.default_system_prompts = [
            "Siz STAT AI — O‘zbekiston Respublikasi Davlat statistika qo‘mitasining sun’iy intellektli chatbotisiz.",
            "Siz faqat o‘zbek tilida, imlo va grammatik xatolarsiz javob berishingiz kerak.",
            "Siz faqat Davlat statistika qo‘mitasi faoliyatiga oid savollarga javob bera olasiz.",
            "Siz hech qanday yangi ma’lumot o‘ylab topmasligingiz kerak — faqat sizga berilgan ma’lumotlar asosida javob bering.",
            "Siz faqat ko‘chmas mulk, rasmiy statistik hisobotlar va “ESTAT 4.0” tizimiga oid savollarga javob bera olasiz.",
            "Siz faqat Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobot shakllari doirasidagi savollarga javob berishingiz mumkin.",
            "Barcha javoblaringizni faqat HTML formatda quyidagi teglar yordamida chiqaring: <b>, <i>, <p>. Markdown yoki LaTeXdan foydalanmang."
        ]

        # Har bir session uchun alohida chat tarixi
        self.session_histories: Dict[str, List[BaseMessage]] = defaultdict(list)

    def _create_messages(
        self,
        system_prompts: List[str],
        user_prompt: str,
        session_id: Optional[str] = None
    ) -> List[BaseMessage]:
        messages = []

        # System promptlar
        for prompt in system_prompts:
            messages.append(SystemMessage(content=prompt))

        # Agar tarix mavjud bo‘lsa, qo‘shamiz
        if session_id and session_id in self.session_histories:
            messages += self.session_histories[session_id]

        # Hozirgi so‘rov
        messages.append(HumanMessage(content=user_prompt))

        return messages

    async def chat(self, prompt: str, session_id: str) -> str:
        """Session asosida kontekst bilan chat qilish"""
        messages1 = self._create_messages(self.default_system_prompts, prompt, session_id)
        messages2 = self._create_messages(
            self.default_system_prompts,
            prompt + ". Estat 4.0 va hisobot shakillari",
            session_id
        )

        error_system_prompt = "If there is no relevant information in the context, respond with: '<p>Ma'lumot bazamdan ushbu savol bo'yicha ma'lumot topilmadi. Iltimos, boshqa savol berishingiz mumkin.</p>'."
        error_system_prompts = self.default_system_prompts + [error_system_prompt]
        messages3 = self._create_messages(error_system_prompts, prompt, session_id)

        responses = await asyncio.gather(
            self._invoke(messages1),
            self._invoke(messages2),
            self._invoke(messages3)
        )

        all_responses = "\n".join(responses)

        merge_messages = [
            SystemMessage(content="Siz quyidagi 3 ta model javobini o'qing va ularni yagona, izchil va aniq HTML formatdagi javobga birlashtiring. Faqat <p></p>, <b></b>, <i></i> teglaridan foydalaning. Yagona javobni qaytaring."),
            HumanMessage(content=all_responses)
        ]

        merged = await self._invoke(merge_messages)

        # ====> Tarixga qo‘shish
        self.session_histories[session_id].append(HumanMessage(content=prompt))
        self.session_histories[session_id].append(AIMessage(content=merged))

        # Tarixni cheklash (20 xabardan ortig‘ini o‘chir)
        if len(self.session_histories[session_id]) > 20:
            self.session_histories[session_id] = self.session_histories[session_id][-20:]

        return merged

    async def _invoke(self, messages: List[BaseMessage]) -> str:
        try:
            result = await self.model.ainvoke(messages)
            cleaned_content = result.content.replace("```html", "").replace("```", "")
            return cleaned_content
        except Exception as e:
            print("Error in invoke:", str(e))
            raise e

    async def rewrite_query(self, user_query: str, chat_history: str) -> dict:
        try:
            # if(len(chat_history)):
            #     return user_query

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

    
    async def generate_questions(self, question: str) -> list:
        prompt = f"""Quyidagi savolga 3 ta sinonim shaklida savol yozing. Har bir sinonim asl savol bilan bir xil ma'noni bildirishi kerak, lekin turlicha ifodalangan bo'lsin. Keraksiz yoki qo'shimcha ma'no qo'shmang.
                    Ma'lumotni list ko'rinishida ber
                    Salomlashish va shunga o'xshagan savollar bo'lsa bosh array qaytar
                    savollarni boshiga nmer qo'ymang
                Savol: {question}"""

        try:
            messages = [HumanMessage(content=prompt)]
            result = await self.model.ainvoke(messages)
            sentences = [s.strip() for s in result.content.split('\n') if s.strip()]
            return sentences[:3]
        except Exception as e:
            print(f"Error in generating sentences: {str(e)}")
            return ["Error generating sentences.", "Please try again.", "Check your connection."]


# === Model instance va funksiyalar ===
model = LangChainOllamaModel()

async def chat(prompt: str, session_id: str) -> str:
    return await model.chat(prompt, session_id)

async def rewrite_query(user_query: str, chat_history: str) -> dict:
    return await model.rewrite_query(user_query, chat_history)

async def generate_questions(question: str) -> list:
    return await model.generate_questions(question)