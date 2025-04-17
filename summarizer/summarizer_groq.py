import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain importlari
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_groq import ChatGroq

# .env faylidan API kalitni o'qish
load_dotenv()

class GroqSummarizer:
    """
    Groq LLM orqali matnni qisqartiruvchi klass.
    Matnlarni qisqartirish va bir nechta matnlarni birlashtirib qisqartirish funksionalligini taqdim etadi.
    """
    
    def __init__(self, model_name: str = "llama3-70b-8192"):
        """
        Groq summarizer-ni ishga tushirish.
        
        Args:
            model_name: Groq LLM modeli nomi.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model_name = model_name
        
        # LangChain ChatGroq modelini yaratish
        self.model = ChatGroq(
            model_name=self.model_name,
            groq_api_key=self.api_key,
            temperature=0.3,  # Qisqartirish uchun past harorat kerak - aniqroq natija beradi
            max_tokens=1000    # Qisqartmalar uchun etarli uzunlik
        )
        
        # Qisqartirish uchun asosiy system prompt
        self.summarize_system_prompt = """Siz qisqartiruvchi yordamchisiz. Sizning vazifangiz matnlarni qisqa, lekin asosiy ma'lumotlarni o'z ichiga olgan tarzda qisqartirish. 
        Faqat berilgan matndan ma'lumot oling va o'zingizdan hech narsa qo'shmang.
        Qisqartmada matnning eng muhim qismlarini saqlang va ortiqcha tafsilotlarni olib tashlang.
        """
    
    async def _invoke_model(self, text: str, max_length: int = 512) -> str:
        """
        Groq modelini chaqirish va qisqartmani olish uchun yordamchi metod.
        
        Args:
            text: Qisqartirilishi kerak bo'lgan matn.
            max_length: Qisqartmaning maksimal uzunligi.
            
        Returns:
            Qisqartirilgan matn.
        """
        messages = [
            SystemMessage(content=f"{self.summarize_system_prompt}\nQisqartmani {max_length} so'zdan oshirmaslikka harakat qiling."),
            HumanMessage(content=f"Quyidagi matnni qisqartiring: {text}")
        ]
        
        response = await self.model.ainvoke(messages)
        return response.content
    
    async def summarize(self, text: str, max_length: int = 512) -> str:
        """
        Berilgan matnni qisqartirish.
        
        Args:
            text: Qisqartirilishi kerak bo'lgan matn.
            max_length: Qisqartmaning maksimal uzunligi.
            
        Returns:
            Qisqartirilgan matn.
        """
        if not text.strip():
            return ""
            
        return await self._invoke_model(text, max_length)
    
    async def batch_summarize(self, texts: List[str], max_length: int = 512) -> List[str]:
        """
        Bir nechta matnlarni parallel ravishda qisqartirish.
        
        Args:
            texts: Qisqartirilishi kerak bo'lgan matnlar ro'yxati.
            max_length: Har bir qisqartmaning maksimal uzunligi.
            
        Returns:
            Qisqartirilgan matnlar ro'yxati.
        """
        if not texts:
            return []
            
        # Barcha matnlarni parallel ravishda qisqartirish
        tasks = [self._invoke_model(text, max_length) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def summarize_chunks(self, chunks: List[str], max_chunk_length: int = 512,
                             max_combined_length: int = 512) -> str:
        """
        Ro'yxatdagi har bir matn bo'lagini qisqartirish, keyin barcha qisqartmalarni 
        birlashtirib, yakuniy qisqartmani yaratish.
        
        Args:
            chunks: Qisqartirilishi kerak bo'lgan matn bo'laklari ro'yxati.
            max_chunk_length: Har bir bo'lak qisqartmasining maksimal uzunligi.
            max_combined_length: Yakuniy qisqartmaning maksimal uzunligi.
            
        Returns:
            Barcha bo'laklarning yakuniy birlashtirilgan qisqartmasi.
        """
        if not chunks:
            return ""
            
        # Avval har bir bo'lakni alohida qisqartirish
        chunk_summaries = await self.batch_summarize(chunks, max_chunk_length)
        
        # Qisqartmalarni birlashtirib, bir matn hosil qilish
        combined_text = "\n".join(chunk_summaries)
        
        # Birlashtirilgan matnni yakuniy qisqartirish
        # final_summary = await self.summarize(combined_text, max_combined_length)
        
        # return final_summary
        return combined_text
    
    async def process_text_chunks(self, chunks_list: List[str], **kwargs) -> str:
        """
        Matn bo'laklari ro'yxatini qayta ishlash, har birini qisqartirish va birlashtirib yakuniy qisqartma yaratish.
        Bu asinxron summarize_chunks metodini to'g'ridan-to'g'ri ishlatish uchun asynchronous wrapper metod.
        
        Args:
            chunks_list: Qayta ishlanishi kerak bo'lgan matn bo'laklari ro'yxati.
            **kwargs: summarize_chunks metodiga uzatiladigan qo'shimcha parametrlar.
            
        Returns:
            Barcha matn bo'laklarining birlashtirilgan qisqartmasi.
        """
        # Chunki bu metod o'zi async, to'g'ridan-to'g'ri summarize_chunks ni chaqiramiz
        try:
            # To'g'ridan-to'g'ri asinxron metodini chaqirish
            result = await self.summarize_chunks(chunks_list, **kwargs)
            return result
        except Exception as e:
            # Xatolarni qayta ko'tarish
            print(f"Error in summarize_chunks: {str(e)}")
            raise e

    async def summarize_large_text(self, text: str, max_chunk_length: int = 512, max_combined_length: int = 512) -> str:
        """
        Katta matnni bo'laklarga bo'lib, qisqartirish va birlashtirib yakuniy qisqartma yaratish.
        
        Args:
            text: Katta matn.
            max_chunk_length: Har bir bo'lak qisqartmasining maksimal uzunligi.
            max_combined_length: Yakuniy qisqartmaning maksimal uzunligi.
            
        Returns:
            Barcha bo'laklarning yakuniy birlashtirilgan qisqartmasi.
        """
        # Matnni bo'laklarga bo'lish
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        # Bo'laklarni qisqartirish va birlashtirish
        return await self.summarize_chunks(chunks, max_chunk_length, max_combined_length)

groq_summarizer = GroqSummarizer()
# combined_summary = groq_summarizer.process_text_chunks(chunks)
# print(combined_summary, "<- combined_summary")
# Misol uchun kod
# if __name__ == "__main__":
#     # Groq summarizer-ni ishga tushirish
#     summarizer = GroqSummarizer()
    
#     # Misol 1: Bitta matnni qisqartirish
#     text = """
#     Sun'iy intellekt (AI) - bu odamlar va hayvonlar ko'rsatadigan tabiiy intellektdan farqli o'laroq, mashinalar tomonidan namoyish etiladigan intellektdir.
#     AI tadqiqotlari intellektual agentlarni o'rganish sohasi sifatida ta'riflangan bo'lib, bu har qanday tizimga ishora qiladi, u o'z muhitini idrok etadi va o'z maqsadlariga erishish imkoniyatini maksimal darajada oshiradigan harakatlarni amalga oshiradi.
#     "Sun'iy intellekt" atamasi avval "o'rganish" va "muammolarni hal qilish" kabi inson miyasi bilan bog'liq bo'lgan "odam" kognitiv ko'nikmalarini taqlid qiladigan va ko'rsatadigan mashinalarni tasvirlash uchun ishlatilgan.
#     Bu ta'rif o'zidan keyin AI bo'yicha etakchi tadqiqotchilar tomonidan rad etilgan, ular endi AI ni ratsionallik va oqilona harakat qilish nuqtai nazaridan tasvirlaydilar, bu intellektni qanday ifodalash mumkinligini cheklamaydi.
#     """
    
#     # Matnni qisqartirish
#     summary = summarizer.process_text_chunks([text])
#     print("Bitta matn qisqartmasi:\n", summary)
    
#     # Misol 2: Matn bo'laklari ro'yxatini qisqartirish
#     chunks = [
#         "Sun'iy intellekt (AI) - bu odamlar va hayvonlar ko'rsatadigan tabiiy intellektdan farqli o'laroq, mashinalar tomonidan namoyish etiladigan intellektdir.",
#         "AI ilovalari rivojlangan veb-qidiruv tizimlari, tavsiya tizimlari (YouTube, Amazon va Netflix tomonidan foydalaniladigan), inson nutqini tushunish (Siri va Alexa kabi), o'z-o'zini boshqaradigan mashinalar (masalan, Waymo), generativ yoki ijodiy vositalar (ChatGPT va AI san'at), avtomatlashtirilgan qaror qabul qilish va strategik o'yin tizimlarida (shaxmat va Go kabi) eng yuqori darajada raqobatlashishni o'z ichiga oladi.",
#         "Mashinalar tobora qobiliyatli bo'lib borgan sari, 'intellektni' talab qiladi deb hisoblanadigan vazifalar ko'pincha AI ta'rifidan chiqarib tashlanadi, bu AI effekti deb ataladigan hodisa.",
#         "Masalan, optik belgilarni tanib olish ko'pincha AI deb hisoblanadigan narsalardan chiqarib tashlanadi, chunki u oddiy texnologiyaga aylanib qolgan."
#     ]
    
#     # Bo'laklarni qayta ishlash
#     combined_summary = summarizer.process_text_chunks(chunks)
#     print(combined_summary, "<- combined_summary")
#     print("\nBo'laklardan birlashtirilgan qisqartma:\n", combined_summary)