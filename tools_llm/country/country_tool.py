import json
import os
import re
import torch
import uuid
import logging
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import util
from retriever.langchain_chroma import CustomEmbeddingFunction

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CountryTool(BaseTool):
    """Davlatlar ma'lumotlarini qidirish uchun tool."""
    name: str = "country_tool"
    description: str = "Davlatlar ma'lumotlarini qidirish uchun mo'ljallangan vosita. Bu tool orqali davlatlarning qisqa nomi, to'liq nomi, harf kodi va raqamli kodi bo'yicha qidiruv qilish mumkin. Misol uchun: \"AQSH\", \"Rossiya\", \"UZ\", \"398\" (Qozog'iston raqamli kodi) kabi so'rovlar orqali ma'lumotlarni izlash mumkin. Natijalar davlat kodi, qisqa nomi, to'liq nomi va kodlari bilan qaytariladi."""
    
    country_data: List[Dict] = Field(default_factory=list)
    use_embeddings: bool = Field(default=False)
    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)
    
    def __init__(self, country_file_path: str, use_embeddings: bool = True, embedding_model=None):
        """Initialize the Country tool with the path to the Country JSON file."""
        super().__init__()
        self.country_data = self._load_country_data(country_file_path)
        self.use_embeddings = use_embeddings
        self.entity_embeddings_cache = None
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logging.info("Tashqaridan berilgan embedding modeli o'rnatildi.")
        else:
            # Embedding modelini kerak bo'lganda yuklaymiz
            self.embedding_model = None
            
        # Embedding ma'lumotlarini tayyorlash
        self._prepare_embedding_data()
    
    def _load_country_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load Country data from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Country data file not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # So'rovni qayta ishlash va Country ma'lumotlaridan tegishli ma'lumotni topish
            result = self._search_country_data(query)
            if not result:
                return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Agar natijada "Davlat ma'lumotlari" qismi bo'lmasa, qo'shish
            if not result.startswith("Davlat ma'lumotlari"):
                result = f"Davlat ma'lumotlari:\n\n{result}"
            
            # Natijani aniq formatda qaytarish
            return result
        except Exception as e:
            logging.error(f"Davlat ma'lumotlarini qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"
    
    def _prepare_embedding_data(self):
        """Embedding uchun ma'lumotlarni tayyorlash"""
        self.entity_texts = []
        self.entity_infos = []
        
        # Davlatlar ma'lumotlarini qo'shish
        for country in self.country_data:
            country_text = f"{country.get('shortName', '')} {country.get('FullName', '')} {country.get('LetterCode', '')} {country.get('DigitalCode', '')}"
            self.entity_texts.append(country_text)
            self.entity_infos.append(("country", country))
        
        # Embeddinglarni kerak bo'lganda hisoblaymiz
        try:
            if self.use_embeddings and self.entity_texts:
                logging.info(f"Davlatlar uchun {len(self.entity_texts)} ta embedding ma'lumoti tayyorlandi.")
                # Embeddinglarni kerak bo'lganda hisoblaymiz
                self.entity_embeddings_cache = None
        except Exception as e:
            logging.error(f"Davlatlar embeddinglarini tayyorlashda xatolik: {str(e)}")
    
    def _search_country_data(self, query: str) -> str:
        """Davlat ma'lumotlaridan so'rov bo'yicha ma'lumot qidirish."""
        try:
            # Agar so'rov bo'sh bo'lsa, umumiy ma'lumot qaytarish
            if not query or query.strip() == "":
                return self._get_countries_list()
            
            # So'rovni tayyorlash
            query = query.strip()
            original_query = query
            query = query.upper()
            
            # Raqamli kod bo'yicha qidirish
            if re.match(r'^\d+$', query):
                for country in self.country_data:
                    if country.get('DigitalCode') == query:
                        return self._format_country_info(country)
            
            # Harf kodi bo'yicha qidirish (2 harfli kod)
            if len(query) == 2:
                for country in self.country_data:
                    if country.get('LetterCode') == query:
                        return self._format_country_info(country)
            
            # Nomi bo'yicha qidirish
            matching_countries = []
            for country in self.country_data:
                short_name = country.get('shortName', '').upper()
                full_name = country.get('FullName', '').upper()
                
                if query in short_name or query in full_name:
                    matching_countries.append(country)
            
            if matching_countries:
                if len(matching_countries) == 1:
                    return self._format_country_info(matching_countries[0])
                else:
                    result = "Davlat ma'lumotlari:\n\n"
                    for country in matching_countries:
                        result += self._format_country_info(country) + "\n\n"
                    return result
            
            # Semantik qidirish
            if self.use_embeddings and self.embedding_model is not None:
                semantic_result = self._semantic_search(original_query)
                if semantic_result:
                    return semantic_result
            
            return f"'{original_query}' so'rovi bo'yicha davlat ma'lumoti topilmadi."
                
        except Exception as e:
            logging.error(f"Davlat ma'lumotlarini qidirishda xatolik: {str(e)}")
            return f"Qidirishda xatolik yuz berdi: {str(e)}"
    
    def _semantic_search(self, query: str) -> str:
        """Semantik qidirish orqali davlat ma'lumotlarini topish"""
        if not self.entity_texts or not self.entity_infos:
            return "Ma'lumotlar mavjud emas."
        
        if not self.use_embeddings:
            return "Semantik qidiruv o'chirilgan."
        
        try:
            # Embedding modelini kerak bo'lganda yuklash (lazy loading)
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                try:
                    self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
                    logging.info("Embedding modeli yuklandi")
                except Exception as e:
                    logging.error(f"Embedding modelini yuklashda xatolik: {str(e)}")
                    return "Embedding modelini yuklashda xatolik."
            
            # Device ni aniqlash (CPU yoki CUDA)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # So'rovni tozalash
            clean_query = query.lower()
            
            # So'rov embeddingini olish va device ga joylashtirish
            model = self.embedding_model.model
            batch_size = 100  # Bir vaqtda qayta ishlanadigan hujjatlar soni
            top_k = 5  # Eng yuqori o'xshashlikdagi natijalar soni
            
            query_embedding = model.encode(clean_query, convert_to_tensor=True, show_progress_bar=False)
            if hasattr(query_embedding, 'to'):
                query_embedding = query_embedding.to(device)
            
            # Barcha ma'lumotlarni embedding qilish
            # Agar entity_embeddings_cache mavjud bo'lmasa, yangi embeddinglarni hisoblash
            if not hasattr(self, 'entity_embeddings_cache') or self.entity_embeddings_cache is None:
                logging.info("Davlatlar embeddinglar keshi yaratilmoqda...")
                self.entity_embeddings_cache = torch.zeros((len(self.entity_texts), model.get_sentence_embedding_dimension()), device=device)
                
                # Hujjatlarni batch usulida qayta ishlash
                for i in range(0, len(self.entity_texts), batch_size):
                    batch_texts = self.entity_texts[i:i+batch_size]
                    
                    # Batch embeddinglarini olish va device ga joylashtirish
                    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    if hasattr(batch_embeddings, 'to'):
                        batch_embeddings = batch_embeddings.to(device)
                    
                    # Embeddinglarni keshga saqlash
                    self.entity_embeddings_cache[i:i+len(batch_texts)] = batch_embeddings
                
                logging.info(f"Davlatlar embeddinglar keshi yaratildi: {len(self.entity_texts)} ta element")
            else:
                # Agar mavjud cache boshqa device da bo'lsa, uni to'g'ri device ga ko'chirish
                if hasattr(self.entity_embeddings_cache, 'device') and self.entity_embeddings_cache.device != device:
                    self.entity_embeddings_cache = self.entity_embeddings_cache.to(device)
            
            # Eng o'xshash ma'lumotlarni topish - device consistency bilan
            cos_scores = torch.cosine_similarity(query_embedding.unsqueeze(0), self.entity_embeddings_cache, dim=1)
            
            # Top natijalarni olish
            top_k = min(top_k, len(cos_scores))
            if top_k == 0:
                return ""
                
            top_results = torch.topk(cos_scores, k=top_k)
            
            # Natijalarni formatlash
            results = []
            for score, idx in zip(top_results[0], top_results[1]):
                if score > 0.5:  # Minimal o'xshashlik chegarasi
                    entity_type, entity_info = self.entity_infos[idx]
                    if entity_type == "country":
                        results.append((score.item(), self._format_country_info(entity_info)))
            
            # Natijalarni o'xshashlik darajasi bo'yicha saralash
            results.sort(key=lambda x: x[0], reverse=True)
            
            if results:
                if len(results) == 1:
                    return results[0][1]
                else:
                    result = "Davlat ma'lumotlari:\n\n"
                    for score, country_info in results:
                        result += f"{country_info} (o'xshashlik: {score:.2f})\n\n"
                    return result
            return ""
        except Exception as e:
            logging.error(f"Semantik qidirishda xatolik: {str(e)}")
            # Xatolik bo'lganda oddiy qidiruv natijasini qaytarish
            matching_countries = []
            query_upper = query.upper()
            for country in self.country_data:
                short_name = country.get('shortName', '').upper()
                full_name = country.get('FullName', '').upper()
                
                if query_upper in short_name or query_upper in full_name:
                    matching_countries.append(country)
            
            if matching_countries:
                if len(matching_countries) == 1:
                    return self._format_country_info(matching_countries[0])
                else:
                    result = "Davlat ma'lumotlari:\n\n"
                    for country in matching_countries[:3]:  # Faqat birinchi 3 tasini ko'rsatish
                        result += self._format_country_info(country) + "\n\n"
                    return result
        
        return ""
    
    def _format_country_info(self, country: Dict) -> str:
        """Davlat ma'lumotlarini formatlash"""
        info = f"ID: {country.get('Id', 'Mavjud emas')}\n"
        info += f"Qisqa nomi: {country.get('shortName', 'Mavjud emas')}\n"
        info += f"To'liq nomi: {country.get('FullName', 'Mavjud emas')}\n"
        info += f"Harf kodi: {country.get('LetterCode', 'Mavjud emas')}\n"
        info += f"Raqamli kodi: {country.get('DigitalCode', 'Mavjud emas')}"
        
        return info
    
    def _get_countries_list(self) -> str:
        """Barcha davlatlar ro'yxatini olish"""
        if not self.country_data:
            return "Davlatlar ro'yxati topilmadi."
        
        result = "Davlatlar ro'yxati:\n\n"
        for i, country in enumerate(self.country_data[:20], 1):  # Faqat birinchi 20 ta davlatni ko'rsatish
            result += f"{i}. {country.get('shortName', 'Mavjud emas')} ({country.get('LetterCode', 'Mavjud emas')})\n"
        
        if len(self.country_data) > 20:
            result += f"\n...va yana {len(self.country_data) - 20} ta davlat mavjud."
        
        return result