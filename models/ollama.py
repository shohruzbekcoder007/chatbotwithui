import aiohttp

class OllamaModel:
    def __init__(self):
        self.base_url = 'http://localhost:11434'
        # self.model = 'gemma2:9b'
        self.model = 'gemma3:12b'
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
            {"role": "system", "content": "Siz milliy statistika qo'mitasining savollariga javob beruvchi chatboti hisoblanasiz."},
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
                'prompt': f"{prompt}\nUser: {user}\nAssistant: {assistant}",
                'stream': False
            }
            
            # So'rovni yuborish
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {'content': result['response']}
                    else:
                        error_text = await response.text()
                        raise Exception(f"Error from Ollama API: {error_text}")
        except Exception as e:
            print("Error in invoke:", str(e))
            raise e

# Model obyektini yaratish
model = OllamaModel()
