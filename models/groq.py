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
        # self.model = "gemma2-9b-it"  # Groq taqdim etgan model
        # self.model = "qwen-2.5-32b"  # Groq taqdim etgan model
        # self.model = "llama-3.1-8b-instant"  # Groq taqdim etgan model
        # self.model = "llama-guard-3-8b"  # Groq taqdim etgan model
        self.model = "llama3-70b-8192"
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
            {"role": "system", "content": "Let the information be based primarily on context and relegate additional answers to a secondary level."},
            {"role": "system", "content": "Do not include unrelated context and present it as separate information."},
            {"role": "system", "content": "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI."},
            {"role": "system", "content": "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan."},
            {"role": "system", "content": "You must respond only in Uzbek. Ensure there are no spelling mistakes."},
            {"role": "system", "content": "You should generate responses strictly based on the given prompt information without creating new content on your own."},
            {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. "},
            {"role": "system", "content": "The questions are within the scope of the estate and reports."},
            # {"role": "system", "content": "The questions are within the scope of the “ESTAT 4.0” and reports. Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobotlar shakillari doirasida"},
            # {"role": "system", "content": "If there is no relevant information in the context, politely try to answer based on your knowledge. If you cannot find the answer or the answer is too far from statistical, reply: '<p>Kechirasiz, ushbu savol bo'yicha aniq ma'lumot topa olmadim. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},
            {"role": "system", "content": "Faqat o'zbek tilida javob ber."},
            {"role": "system", "content": "Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma"},
            {"role": "user", "content": prompt}
        ]

        messages1 = [
            {"role": "system", "content": "Let the information be based primarily on context and relegate additional answers to a secondary level."},
            {"role": "system", "content": "Do not include unrelated context and present it as separate information."},
            {"role": "system", "content": "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI."},
            {"role": "system", "content": "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan."},
            {"role": "system", "content": "You must respond only in Uzbek. Ensure there are no spelling mistakes."},
            {"role": "system", "content": "You should generate responses strictly based on the given prompt information without creating new content on your own."},
            {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. "},
            {"role": "system", "content": "The questions are within the scope of the estate and reports."},
            # {"role": "system", "content": "The questions are within the scope of the “ESTAT 4.0” and reports. Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobotlar shakillari doirasida"},
            # {"role": "system", "content": "If there is no relevant information in the context, politely try to answer based on your knowledge. If you cannot find the answer or the answer is too far from statistical, reply: '<p>Kechirasiz, ushbu savol bo'yicha aniq ma'lumot topa olmadim. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            {"role": "system", "content": "Faqat o'zbek tilida javob ber."},
            {"role": "system", "content": "Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma"},
            {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},
            {"role": "user", "content": prompt + ". Estat 4.0 va hisobot shakillari"}
        ]

        messages2 = [
            {"role": "system", "content": "Let the information be based primarily on context and relegate additional answers to a secondary level."},
            {"role": "system", "content": "Do not include unrelated context and present it as separate information."},
            {"role": "system", "content": "You are a chatbot answering questions for the National Statistics Committee. Your name is STAT AI."},
            {"role": "system", "content": "You are an AI assistant agent of the National Statistics Committee of the Republic of Uzbekistan."},
            {"role": "system", "content": "You must respond only in Uzbek. Ensure there are no spelling mistakes."},
            {"role": "system", "content": "You should generate responses strictly based on the given prompt information without creating new content on your own."},
            {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. "},
            {"role": "system", "content": "The questions are within the scope of the estate and reports."},
            # {"role": "system", "content": "The questions are within the scope of the “ESTAT 4.0” and reports. Davlat statistika hisobotlarini elektron shaklda taqdim etishning avtomatlashtirilgan axborot tizimi va hisobotlar shakillari doirasida"},
            {"role": "system", "content": "Faqat o'zbek tilida javob ber."},
            {"role": "system", "content": "Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma"},
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

        new_response = await self.logical_context("\n".join([r["content"] for r in responses]))

        print(new_response, "<-new_response")

        return new_response['content']

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
                {"role": "system", "content": "Let the information be based primarily on context and relegate additional answers to a secondary level."},
                {"role": "system", "content": "Do not include unrelated context and present it as separate information."},
                {"role": "system", "content": "Given the conversation history, rewrite the user query to be fully self-contained and short. Give response in Uzbek. Only give the user's ask."},
                {"role": "user", "content": f"Chat history:\n{chat_history}\n\nUser query:\n{user_query}"}
            ]
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
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

    async def logical_context(self, text: str):
        try:
            messages = [
                {"role": "system", "content": "Let the information be based primarily on context and relegate additional answers to a secondary level."},
                {"role": "system", "content": "Don't add unrelated context"},
                {"role": "system", "content": f"From the following text: 1. Remove sentences that are not consistent in content (irrelevant or off-topic), 2. Remove repeated  sentences. Don't mix up the sentences. Rewrite the text in a logically consistent and simplified way: TEXT:{text}"},
                {"role": "system", "content": "Please provide the answer in Uzbek."},
                {"role": "system", "content": "Faqat o'zbek tilida javob ber."},
                {"role": "system", "content": "Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma"},
                {"role": "system", "content": "Use the same meaning only once."},
                {"role": "system", "content": "Kontextda mavjud bo'lmagan ma'lumotlarni qo'shma"},
                {"role": "system", "content": "Bir xil manodagi gaplar bo'lsa ularni qisqartir"},
                {"role": "system", "content": "I will not answer you, the answer is as follows, without words, the answer is self-replying"},
                {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},
            ]
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
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
        
    async def generate_questions(self, question: str) -> list:
        prompt = f"""Quyidagi savolga 3 ta sinonim shaklida savol yozing. Har bir sinonim asl savol bilan bir xil ma'noni bildirishi kerak, lekin turlicha ifodalangan bo‘lsin. Keraksiz yoki qo‘shimcha ma'no qo‘shmang.
                        Ma'lumotni list ko'rinishida ber
                        Salomlashish va shunga o'xshagan savollar bo'lsa bosh array qaytar
                        savollarni boshiga nmer qo'ymang
                    Savol: {question}"""
        
        try:
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        response = result['choices'][0]['message']['content']
                        sentences = [s.strip() for s in response.split('\n') if s.strip()]
                        return sentences[:3]  # Ensure we return exactly 3 sentences
                    else:
                        error_text = await response.text()
                        raise Exception(f"Error from Groq API: {error_text}")
        except Exception as e:
            print(f"Error in generating sentences: {str(e)}")
            return ["Error generating sentences.", "Please try again.", "Check your connection."]
        
# Model obyektini yaratish
model = GroqModel()