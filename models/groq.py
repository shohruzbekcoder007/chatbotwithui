import os
import aiohttp
from dotenv import load_dotenv

# .env faylidan API kalitni o'qish
load_dotenv()

class GroqModel:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        # self.model = "gemma2-9b-it"  # Groq taqdim etgan model
        self.model = "qwen-2.5-32b"  # Groq taqdim etgan model
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
            # {"role": "system", "content": "If there is no relevant information in the context, politely try to answer based on your knowledge. If you cannot find an answer, respond with: '<p>Kechirasiz, ushbu savol bo'yicha aniq ma'lumot topa olmadim. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            {"role": "user", "content": prompt}
        ]

        response = await self.invoke(messages)
        return response['content']

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