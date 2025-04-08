import os
import aiohttp
from dotenv import load_dotenv
import asyncio

# .env faylidan API kalitni o'qish
load_dotenv()

class GroqModel:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "gemma2-9b-it"  # Groq taqdim etgan model
        # self.model = "qwen-2.5-32b"  # Groq taqdim etgan model
        # self.model = "llama-3.1-8b-instant"  # Groq taqdim etgan model
        # self.model = "llama-guard-3-8b"  # Groq taqdim etgan model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def chat(self, prompt: str) -> str:
        """
        Modelga savol yuborish va javob olish

        Args:
            prompt (str): Savol matni

        Returns:
            str: Model javobi
        """
        messages = [
            {"role": "system", "content": "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI."},
            {"role": "system", "content": "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan."},
            {"role": "system", "content": "You must respond only in Uzbek. Ensure there are no spelling mistakes."},
            {"role": "system", "content": "You should generate responses strictly based on the given prompt information without creating new content on your own."},
            {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. "},
            {"role": "system", "content": "The questions are within the scope of the estate and reports."},
            {"role": "system", "content": "The questions are within the scope of the “ESTAT 4.0” and reports. Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobotlar shakillari doirasida"},
            # {"role": "system", "content": "If there is no relevant information in the context, politely try to answer based on your knowledge. If you cannot find the answer or the answer is too far from statistical, reply: '<p>Kechirasiz, ushbu savol bo'yicha aniq ma'lumot topa olmadim. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},
            {"role": "user", "content": prompt}
        ]

        messages1 = [
            {"role": "system", "content": "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI."},
            {"role": "system", "content": "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan."},
            {"role": "system", "content": "You must respond only in Uzbek. Ensure there are no spelling mistakes."},
            {"role": "system", "content": "You should generate responses strictly based on the given prompt information without creating new content on your own."},
            {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. "},
            {"role": "system", "content": "The questions are within the scope of the estate and reports."},
            {"role": "system", "content": "The questions are within the scope of the “ESTAT 4.0” and reports. Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobotlar shakillari doirasida"},
            # {"role": "system", "content": "If there is no relevant information in the context, politely try to answer based on your knowledge. If you cannot find the answer or the answer is too far from statistical, reply: '<p>Kechirasiz, ushbu savol bo'yicha aniq ma'lumot topa olmadim. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},
            {"role": "user", "content": prompt + ". Estat 4.0 va hisobot shakillari"}
        ]

        messages2 = [
            {"role": "system", "content": "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI."},
            {"role": "system", "content": "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan."},
            {"role": "system", "content": "You must respond only in Uzbek. Ensure there are no spelling mistakes."},
            {"role": "system", "content": "You should generate responses strictly based on the given prompt information without creating new content on your own."},
            {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. "},
            {"role": "system", "content": "The questions are within the scope of the estate and reports."},
            {"role": "system", "content": "The questions are within the scope of the “ESTAT 4.0” and reports. Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobotlar shakillari doirasida"},
            {"role": "system", "content": "If there is no relevant information in the context, respond with: '<p>Ma'lumot bazamdan ushbu savol bo'yicha ma'lumot topilmadi. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},
            {"role": "user", "content": prompt}
        ]

        # 3 ta javobni parallel chaqirish
        responses = await asyncio.gather(
            self.invoke(messages),
            self.invoke(messages1),
            self.invoke(messages2)
        )

        all_responses = "\n".join([r["content"] for r in responses])

        print("All responses ->", all_responses)

        # Endi ushbu javoblarni birlashtirish uchun boshqa so'rov yuboramiz
        merge_messages = [
            {"role": "system", "content": "Siz quyidagi 3 ta model javobini o'qing va ularni yagona, izchil va aniq HTML formatdagi javobga birlashtiring. Faqat <p></p>, <b></b>, <i></i> teglaridan foydalaning. Yagona javobni qaytaring."},
            {"role": "user", "content": f"{all_responses}"}
        ]

        # Yakuniy birlashtirilgan javobni olish
        merged = await self.invoke(merge_messages)

        return merged["content"]

    async def invoke(self, messages):
        try:
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {'content': result['choices'][0]['message']['content']}
                    else:
                        error_text = await response.text()
                        raise Exception(f"Error from Groq API: {error_text}")
        except Exception as e:
            print("Error in invoke:", str(e))
            raise e

    async def rewrite_query(self, user_query, chat_history):
        try:
            messages = [
                {"role": "system", "content": "Given the conversation history, rewrite the user query to be fully self-contained and short. Give response in Uzbek. Only give the user's ask."},
                {"role": "user", "content": f"Chat history:\n{chat_history}\n\nUser query:\n{user_query}"}
            ]
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 200,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {'content': result['choices'][0]['message']['content']}
                    else:
                        error_text = await response.text()
                        raise Exception(f"Error from Groq API: {error_text}")
        except Exception as e:
            print("Error in rewrite_query:", str(e))
            raise e

# Model obyektini yaratish
model = GroqModel()