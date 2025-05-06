# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# import os
# import torch
# from dotenv import load_dotenv
# from typing import List, Dict, Optional
# from queue import Queue
# from threading import Lock
# import time
# import uuid

# class GemmaModel:
#     _instance = None
#     _lock = Lock()
#     _is_initialized = False
    
#     def __new__(cls):
#         if cls._instance is None:
#             with cls._lock:
#                 if cls._instance is None:
#                     cls._instance = super().__new__(cls)
#         return cls._instance
    
#     def __init__(self):
#         if not GemmaModel._is_initialized:
#             with self._lock:
#                 if not GemmaModel._is_initialized:
#                     self.system_prompt = """Siz O'zbek tilidagi AI yordamchisiz. Savolga aniq va tushunarli javob bering.
#                     Agar savolga javob bera olmasangiz, buni ochiq ayting."""
#                     self._initialize_model()
#                     self.request_queue = Queue()
#                     self.processing = False
#                     GemmaModel._is_initialized = True
    
#     def _initialize_model(self):
#         """Modelni inizializatsiya qilish"""
#         load_dotenv()
#         self.token = os.getenv("HUGGINGFACE_TOKEN")
#         if not self.token:
#             raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
            
#         # GPU tekshirish
#         print("\nGPU Diagnostikasi:")
#         print(f"CUDA mavjudmi: {torch.cuda.is_available()}")
#         if torch.cuda.is_available():
#             print(f"GPU nomi: {torch.cuda.get_device_name(0)}")
#             print(f"CUDA versiyasi: {torch.version.cuda}")
        
#         # 4-bit kvantlash sozlamalari
#         self.quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#         )
        
#         print(" Tokenizer yuklanmoqda...")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "google/gemma-7b-it",
#             token=self.token,
#             trust_remote_code=True
#         )
        
#         print(" Model yuklanmoqda...")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             "google/gemma-7b-it",
#             device_map="auto",
#             token=self.token,
#             trust_remote_code=True,
#             quantization_config=self.quantization_config
#         )
#         print("Model yuklandi va ishlatishga tayyor!")
    
#     async def generate_response(self, messages: List[Dict[str, str]], 
#                         max_tokens: int = 100,
#                         temperature: float = 0.7) -> str:
#         """Chat xabarlariga javob generatsiya qilish"""
#         try:
#             # Oddiy prompt yaratish
#             prompt = ""
#             for msg in messages:
#                 if msg["role"] == "user":
#                     prompt += f"Human: {msg['content']}\n"
#                 else:
#                     prompt += f"Assistant: {msg['content']}\n"
            
#             prompt += "Assistant: "
#             print("Final prompt:", prompt)
            
#             # Input tokenizatsiya
#             inputs = self.tokenizer(
#                 prompt, 
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=2048  # Tokenlar sonini cheklash
#             ).to(self.model.device)
            
#             # Generatsiya
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=max_tokens,
#                 do_sample=True,
#                 temperature=temperature,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 num_beams=1,  # Beam search o'rniga greedy
#                 top_k=50,     # Top-k sampling
#                 top_p=0.9     # Nucleus sampling
#             )
            
#             # Javobni dekodlash
#             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             response = response.replace(prompt, "").strip()
#             return response
            
#         except Exception as e:
#             print(f"Error in generate_response: {str(e)}")
#             if torch.cuda.is_available():
#                 print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
#             return str(e)

#     async def chat(self, context: str, query: str) -> str:
#         """
#         Modelga savol yuborish va javob olish

#         Args:
#             context (str): Kontekst matni
#             query (str): Savol matni

#         Returns:
#             str: Model javobi
#         """
#         try:
#             # System va user promptlaridan xabarlar yaratish
#             messages = self._create_messages(context, query)
#             print("Created messages:", messages)

#             # Modelni chaqirish
#             response = await self.generate_response(messages)
#             print("Got response:", response)
            
#             return response
            
#         except Exception as e:
#             print(f"Error in chat method: {str(e)}")
#             return str(e)

#     def _create_messages(self, context: str, query: str) -> list:
#         """
#         User/Assistant promptlarini yaratish
#         """
#         messages = []
        
#         # System promptni qisqa qo'shish
#         if hasattr(self, 'system_prompt') and self.system_prompt:
#             messages.append({"role": "user", "content": self.system_prompt})
#             messages.append({"role": "assistant", "content": "Tushundim."})
        
#         # Kontekst qo'shish
#         if context and context.strip():
#             messages.append({"role": "user", "content": context})
        
#         # Savolni qo'shish
#         if query and query.strip():
#             messages.append({"role": "user", "content": query})
        
#         return messages
    
#     def start_processing(self):
#         """Queue processing ni boshlash"""
#         import threading
#         if not self.processing:
#             self.processing = True
#             thread = threading.Thread(target=self.process_queue, daemon=True)
#             thread.start()
    
#     def get_queue_size(self) -> int:
#         """Queue dagi so'rovlar soni"""
#         return self.request_queue.qsize()

#     def process_queue(self):
#         """Queue dagi so'rovlarni qayta ishlash"""
#         while True:
#             try:
#                 with self._lock:
#                     if self.request_queue.empty():
#                         continue
                    
#                     request_id, messages, max_tokens, temperature, result = self.request_queue.get()
                    
#                     try:
#                         # Chat formatini qo'llash
#                         prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
#                         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                        
#                         with torch.inference_mode():
#                             outputs = self.model.generate(
#                                 **inputs,
#                                 max_new_tokens=max_tokens,
#                                 do_sample=True,
#                                 temperature=temperature
#                             )
                        
#                         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                         response = response.replace(prompt, "").strip()
                        
#                         result["status"] = "completed"
#                         result["response"] = response
                        
#                     except Exception as e:
#                         result["status"] = "error"
#                         result["error"] = str(e)
#                         print(f"Error in processing request {request_id}: {str(e)}")
                    
#                     self.request_queue.task_done()
                    
#             except Exception as e:
#                 print(f"Queue processing error: {str(e)}")
#                 time.sleep(1)
    
# # # Modeldan foydalanish misoli
# # if __name__ == "__main__":
# #     # Model obyektini yaratish
# #     gemma = GemmaModel()
# #     gemma.start_processing()
    
# #     # Test so'rovlar
# #     messages = [
# #         {"role": "user", "content": "O'zbekiston qayerda joylashgan?"}
# #     ]
    
# #     # Parallel so'rovlar yuborish
# #     import threading
# #     def test_request(user_id):
# #         response = gemma.chat("", messages[0]["content"])
# #         print(f"\nUser {user_id}:")
# #         print(f"Savol: {messages[0]['content']}")
# #         print(f"Javob: {response}")
    
# #     # 3 ta parallel so'rov
# #     threads = []
# #     for i in range(3):
# #         thread = threading.Thread(target=test_request, args=(i,))
# #         threads.append(thread)
# #         thread.start()
    
# #     # Barcha so'rovlar tugashini kutish
# #     for thread in threads:
# #         thread.join()