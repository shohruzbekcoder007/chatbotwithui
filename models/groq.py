# import os
# import aiohttp
# from typing import Dict, Optional

# class GroqModel:
#     def __init__(self, api_key: Optional[str] = None):
#         self.api_key = "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"
#         if not self.api_key:
#             raise ValueError("GROQ_API_KEY must be provided either through constructor or environment variable")
        
#     async def chat(self, prompt: str, query: str) -> Dict:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#         data = {
#             "model": "mixtral-8x7b-32768",  # Groq's Mixtral model
#             "messages": [
#                 {"role": "system", "content": prompt},
#                 {"role": "user", "content": query}
#             ],
#             "temperature": 0.7
#         }

#         async with aiohttp.ClientSession() as session:
#             try:
#                 async with session.post(
#                     "https://api.groq.com/v1/chat/completions", 
#                     json=data, 
#                     headers=headers
#                 ) as response:
#                     if response.status != 200:
#                         error_data = await response.json()
#                         raise Exception(f"Groq API error: {error_data.get('error', {}).get('message', 'Unknown error')}")
#                     return await response.json()
#             except aiohttp.ClientError as e:
#                 raise Exception(f"Network error while calling Groq API: {str(e)}")

# model_groq = GroqModel()