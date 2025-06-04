import json
import os
import re
import torch
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import util
from retriever.langchain_chroma import CustomEmbeddingFunction

# CustomEmbeddingFunction klassi retriever.langchain_chroma modulidan import qilingan

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DBIBTTool(BaseTool):
    """DBIBT ma'lumotlarini qidirish uchun tool."""
    name: str = "dbibt_tool"
    description: str = "DBIBT ma'lumotlari"

    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    dbibt_data: Dict = Field(default_factory=dict)
    entity_DBIBT: List[str] = Field(default_factory=list)
    entity_OKPO: List[str] = Field(default_factory=list)
    entity_INN: List[str] = Field(default_factory=list)
    entity_NAME: List[str] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    uembedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    
    def __init__(self, dbibt_file_path: str, use_embeddings: bool = True):
        """Initialize the SOATO tool with the path to the SOATO JSON file."""
        super().__init__()
        self.dbibt_data = self._load_dbibt_data(dbibt_file_path)
        self.use_embeddings = use_embeddings
        
        # Embedding modelini yaratish
        if self.use_embeddings and self.embedding_model is None:
            print("Embedding modeli yuklanmoqda...")
            self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
            self._prepare_embedding_data()

    def _load_dbibt_data(self, dbibt_file_path: str) -> Dict:
        """DBIBT ma'lumotlarini yuklash"""
        with open(dbibt_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _prepare_embedding_data(self):
        """Prepare embedding data for the DBIBT data."""
        self.entity_DBIBT = [entity['DBIBT'] for entity in self.dbibt_data]
        self.entity_OKPO = [entity['OKPO'] for entity in self.dbibt_data]
        self.entity_INN = [entity['INN'] for entity in self.dbibt_data]
        self.entity_NAME = [entity['NAME'] for entity in self.dbibt_data]

    def _extract_important_words(self, query: str) -> List[str]:
        """So'rovdan muhim so'zlarni ajratib olish
        
        Args:
            query: Qidirilayotgan so'rov
            
        Returns:
            Muhim so'zlar ro'yxati
        """
        # So'zlarni ajratib olish
        words = query.lower().split()
        
        # Stop so'zlarni o'chirish
        stop_words = ["va", "bilan", "uchun", "ham", "lekin", "ammo", "yoki", "agar", "chunki", "shuning", "bo'yicha", "haqida", "qanday", "nima", "qaysi", "nechta", "qanaqa", "qachon", "qayerda", "kim", "nima", "necha"]
        
        # Muhim so'zlarni ajratib olish - uzunligi 3 dan katta va stop so'z bo'lmagan so'zlar
        important_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return important_words

    def _search_by_text(self, query: str, return_raw: bool = False) -> Optional[Dict]:
        """Search for a specific text in the DBIBT data."""

        # Muhim so'zlarni ajratib olish
        important_words = self._extract_important_words(query)
        logging.info(f"Matn qidiruv uchun muhim so'zlar: {important_words}")
        
        # Natijalarni saqlash uchun ro'yxat
        results = []

        for entity in self.dbibt_data:
            if entity['NAME'].lower() == query.lower() or entity['DBIBT'] == query or entity['OKPO'] == query or entity['INN'] == query:
                results.append(entity)

        for entity in self.entity_DBIBT:
            if entity.lower() == query.lower():
                results.append(entity)

        for entity in self.entity_OKPO:
            if entity.lower() == query.lower():
                results.append(entity)

        for entity in self.entity_INN:
            if entity.lower() == query.lower():
                results.append(entity)

        for entity in self.entity_NAME:
            if entity.lower() == query.lower():
                results.append(entity)

        # Natijalarni qaytarish
        if return_raw:
            return results
        
        if results:
            if len(results) > 5:
                result_text = "\n\n".join(results[:5])
                result_text += f"\n\nJami {len(results)} ta natija topildi."
            else:
                result_text = "\n\n".join(results)
            
            return result_text
        
        return "Natija topilmadi."

    def _semantic_search(self, query: str, return_raw: bool = False) -> Union[List[Dict], str]:
        """Semantik qidiruv."""
        try: 
            if not self.entity_DBIBT or not self.entity_OKPO or not self.entity_INN or not self.entity_NAME:
                return [] if return_raw else "Ma'lumotlar mavjud emas."

            if self.embedding_model is None:
                self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
                self._prepare_embedding_data()

            # Query embeddingini olish
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Entity embeddinglarini olish
            entity_embeddings = self.embedding_model.encode(self.dbibt_data)
            
            # Cosine similarityni hisoblash
            cos_scores = []
            for entity_embedding in entity_embeddings:
                cos_score = util.cos_sim([query_embedding], [entity_embedding])[0][0]
                cos_scores.append(cos_score)
            
            # Natijalarni saralash
            cos_scores_tensor = torch.tensor(cos_scores)
            sorted_indices = torch.argsort(cos_scores_tensor, descending=True)
            
            # Top natijalarni olish
            results = [self.dbibt_data[idx] for idx in sorted_indices]

            if return_raw:
                return results
            
            # Natijalarni formatlash
            formatted_results = []
            for i, result in enumerate(results[:5]):  # Top 5 natijalarni ko'rsatish
                formatted_results.append(f"{i+1}. Kod: {result['DBIBT']}, OKPO: {result['OKPO']}, INN: {result['INN']}, Nomi: {result['NAME']}")
            
            formatted_text = "\n".join(formatted_results)
            
            if len(results) > 5:
                formatted_text += f"\n\nJami {len(results)} ta natija topildi."
            
            return formatted_text
                
        except Exception as e:
            logging.error(f"Semantik qidiruvda xatolik: {str(e)}")
            return [] if return_raw else f"Semantik qidiruvda xatolik: {str(e)}"

    def _search_dbibt_data(self, query: str) -> Union[List[Dict], str]:
        """Search for a specific query in the DBIBT data."""
        try:
            if not self.dbibt_data:
                return "DBIBT data yuklanmagan."

            # So'rovni tozalash
            original_query = query
            query = query.lower()

            # Nuqtalarni saqlab qolish, chunki ular nation kodlari uchun muhim
            query = re.sub(r'[\.,]', '', query)
            logging.info(f"Qidirilayotgan so'rov: '{original_query}', tozalangan so'rov: '{query}'")

            # 2. Hibrid qidiruv - matn va semantik qidiruv natijalarini birlashtirish
            all_results = []

            # Matn qidiruv
            logging.info(f"Matn qidiruv uchun muhim so'zlar: {[word for word in query.split() if len(word) > 3]}")
            text_results = self._search_by_text(query, return_raw=True)

            # Semantik qidiruv
            semantic_results = self._semantic_search(query, return_raw=True)

            # Natijalarni birlashtirish - dublikatlarni olib tashlash
            for result in text_results:
                if result not in all_results:
                    all_results.append(result)
            
            for result in semantic_results:
                if result not in all_results:
                    all_results.append(result)

            # Natijalarni muhim so'zlar asosida saralash
            important_words = [word for word in query.split() if len(word) > 3]

            def score_result(result):
                score = 0
                for word in important_words:
                    if word in result['NAME'].lower():
                        score += 1
                return score
            
            all_results.sort(key=score_result, reverse=True)
            
            # Natijalarni formatlash
            formatted_results = []
            for i, result in enumerate(all_results[:5]):  # Top 5 natijalarni ko'rsatish
                formatted_results.append(f"{i+1}. Kod: {result['DBIBT']}, OKPO: {result['OKPO']}, INN: {result['INN']}, Nomi: {result['NAME']}")
            
            formatted_text = "\n".join(formatted_results)
            
            if len(all_results) > 5:
                formatted_text += f"\n\nJami {len(all_results)} ta natija topildi."
            
            return formatted_text
            
        except Exception as e:
            logging.error(f"DBIBT data qidirishda xatolik: {str(e)}")
            return f"Qidirishda xatolik: {str(e)}"

    def _format_results(self, results: List[Dict]) -> str:
        """Format search results.
        
        Args:
            results: Natijalar ro'yxati
            
        Returns:
            Formatlangan natija matni
        """
        if not results:
            return ""
        
        formatted_results = []
        
        # Natijalarni formatlash
        for i, item in enumerate(results[:5]):  # Top 5 natijalarni ko'rsatish
            formatted_results.append(f"{i+1}. Kod: {item['CODE']}, Nomi: {item['NAME']}")
        
        formatted_text = "\n".join(formatted_results)
        
        # Agar natijalar ko'p bo'lsa, qo'shimcha ma'lumot
        if len(results) > 5:
            formatted_text += f"\n\nJami {len(results)} ta natija topildi."
        
        return formatted_text
    
    def run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # So'rovni qayta ishlash va Nation ma'lumotlaridan tegishli ma'lumotni topish
            result = self._search_dbibt_data(query)
            
            # Agar natija bo'sh bo'lsa yoki xatolik haqida xabar bo'lsa
            if not result:
                return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Agar natija string bo'lsa (xatolik xabari)
            if isinstance(result, str):
                return f"DBIBT ma'lumotlari:\n\n{result}"
            
            # Agar natija ro'yxat bo'lsa (lug'atlar ro'yxati)
            if isinstance(result, list):
                formatted_results = self._format_results(result)
                return f"DBIBT ma'lumotlari:\n\n{formatted_results}"
            
            # Boshqa holatlar uchun
            return f"DBIBT ma'lumotlari:\n\n{str(result)}"
            
        except Exception as e:
            logging.error(f"DBIBT qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"
    
    