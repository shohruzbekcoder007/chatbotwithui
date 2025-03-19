import json
import aiohttp

class OllamaModel:
    def __init__(self):
        self.base_url = 'http://localhost:11434'
        self.model = 'gemma2:9b'
        # self.model = 'llama3.2:3b'

    async def invoke(self, messages):
        print('messages', messages)
        try:
            prompt = ''
            user = ''
            assistant = ''
            
            # Xabarlarni qayta ishlash
            for msg in messages:
                if msg['role'] == 'system':
                    prompt = prompt + f" {msg['content']}"
                elif msg['role'] == 'user':
                    user = user + f" {msg['content']}"
                elif msg['role'] == 'assistant':
                    assistant = assistant + f" {msg['content']}"

            print(f"{prompt=}, {user=}, {assistant=}")

            # Ollama API ga so'rov yuborish
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.model,
                        'prompt': (prompt + user).strip(),
                        'stream': False,
                        'options': {
                            'temperature': 0.7,
                            'top_k': 40,
                            'top_p': 0.9,
                            'repeat_penalty': 1.1,
                            'max_tokens': 500
                        }
                    }
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama API xatoligi: {response.status}")
                    
                    text = await response.text()
                    print('Raw response:', text)
                    
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError as e:
                        print('JSON parse error:', e)
                        print('Raw response:', text)
                        raise Exception('Invalid JSON response from Ollama API')

                    return {
                        'content': result.get('response', "Kechirasiz, hozir javob bera olmadim. Iltimos, qayta urinib ko'ring.")
                    }

        except Exception as error:
            print('Ollama modelida xatolik:', error)
            raise error

# Model obyektini yaratish
model = OllamaModel()
