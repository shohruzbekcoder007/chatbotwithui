import aiohttp

class OllamaModel:
    def __init__(self):
        self.base_url = 'http://localhost:11434'
        # self.model = 'gemma2:9b'
        self.model = 'gemma3:27b'
        # self.model = 'llama3.2:3b'

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
            # {"role": "system", "content": "You are only allowed to answer questions related to the National Statistics Committee. If a question is unrelated, respond with: '<p>Kechirasiz, men faqat Oʻzbekiston Respublikasi Milliy statistika qoʻmitasiga oid savollarga javob bera olaman.</p>'."},
            {"role": "system", "content": "If there is no relevant information in the context, politely try to answer based on your knowledge. If you cannot find an answer, respond with: '<p>Kechirasiz, ushbu savol bo‘yicha aniq ma’lumot topa olmadim. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
 
            # {"role": "system", "content": "If there is no relevant information in the context, respond with: '<p>Ma'lumot bazamdan ushbu savol bo'yicha ma'lumot topilmadi. Iltimos, boshqa savol berishingiz mumkin.</p>'."},
            # {"role": "system", "content": "Output the results only in HTML format, no markdown, no latex. (only use this tags: <b></b>, <i></i>, <p></p>)"},

 
            {"role": "user", "content": prompt}
        ]

        response = await self.invoke(messages)
        return response['content']

    async def invoke(self, messages):
        # print('messages', messages)
        try:
            prompt = ''
            user = ''
            assistant = ''
            
            # Xabarlarni qayta ishlash
            for msg in messages:
                if msg['role'] == 'system':
                    prompt = prompt + f" {msg['content']}"
                elif msg['role'] == 'user':
                    user = msg['content']
                elif msg['role'] == 'assistant':
                    assistant = msg['content']
            
            # So'rovni tayyorlash
            data = {
                'model': self.model,
                'prompt': f"System: {prompt}\n\nUser: {user}\n\nAssistant (respond in HTML format tags ) (only use this tags: <b></b>, <i></i>, <p></p>): {assistant}",
                'stream': False
            }
            
            # So'rovni yuborish
            print("asinxron so'rov yuborildi")
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {'content': result['response'].replace("```html", "").replace("```", "")}
                    else:
                        error_text = await response.text()
                        raise Exception(f"Error from Ollama API: {error_text}")
        except Exception as e:
            print("Error in invoke:", str(e))
            raise e

# Model obyektini yaratish
model = OllamaModel()
