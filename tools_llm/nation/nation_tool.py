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

class NationTool(BaseTool):
    """Nation ma'lumotlarini qidirish uchun tool."""
    name: str = "nation_tool"
    description: str = "O'zbekiston Respublikasi millat klassifikatori ma'lumotlarini qidirish uchun tool. Bu tool orqali millat kodi yoki millat nomi bo'yicha qidiruv qilish mumkin. Masalan: '01' (o'zbek), '05' (rus), yoki 'tojik' kabi so'rovlar bilan qidiruv qilish mumkin. Tool millat kodi, nomi va boshqa tegishli ma'lumotlarni qaytaradi."

    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    nation_data: Dict = Field(default_factory=dict)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)
    
    def __init__(self, nation_file_path: str, use_embeddings: bool = True, embedding_model=None):
        """Initialize the SOATO tool with the path to the SOATO JSON file."""
        super().__init__()
        self.nation_data = self._load_nation_data(nation_file_path)
        self.use_embeddings = use_embeddings
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logging.info("Tashqaridan berilgan embedding modeli o'rnatildi.")
        else:
            # Embedding modelini kerak bo'lganda yuklaymiz
            self.embedding_model = None
            
        # Embedding ma'lumotlarini tayyorlash
        self._prepare_embedding_data()

    def _load_nation_data(self, file_path: str) -> Dict[str, Any]:
        """Load nation data from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Nation data file not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_embedding_data(self):
        """Prepare embedding data for the nation data."""
        self.entity_texts = [entity['NAME'] for entity in self.nation_data]
        self.entity_infos = self.nation_data
        
        # Embeddinglarni kerak bo'lganda hisoblaymiz
        self.entity_embeddings_cache = None
        logging.info(f"Embedding uchun {len(self.entity_texts)} ta davlat ma'lumoti tayyorlandi.")

    def _semantic_search(self, query: str, return_raw: bool = False) -> Union[List[Dict], str]:
        """Semantik qidiruv."""
        try:
            if not self.entity_texts or not self.entity_infos:
                return [] if return_raw else "Ma'lumotlar mavjud emas."
            
            if not self.use_embeddings:
                return [] if return_raw else "Semantik qidiruv o'chirilgan."
            
            # Embedding modeli yuklanmagan bo'lsa, yuklash
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
                self._prepare_embedding_data()
            
            # CustomEmbeddingFunction obyektidan SentenceTransformer modelini olish
            model = self.embedding_model.model
            
            # Xotira tejash uchun batch usulida ishlash
            batch_size = 100  # Bir vaqtda qayta ishlanadigan hujjatlar soni
            
            # Query embeddingini olish
            query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            
            # Entity embeddinglarini olish
            if not hasattr(self, 'entity_embeddings_cache') or self.entity_embeddings_cache is None:
                logging.info("Nation embeddinglar keshi yaratilmoqda...")
                
                # Natijalarni saqlash uchun
                all_scores = []
                
                # Hujjatlarni batch usulida qayta ishlash
                for i in range(0, len(self.entity_texts), batch_size):
                    batch_texts = self.entity_texts[i:i+batch_size]
                    
                    # Batch embeddinglarini olish
                    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    
                    # Har bir embedding uchun o'xshashlikni hisoblash
                    for j, emb in enumerate(batch_embeddings):
                        cos_score = util.cos_sim([query_embedding], [emb])[0][0].item()
                        all_scores.append((i+j, cos_score))
                
                # O'xshashlik bo'yicha saralash
                all_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Top natijalarni olish
                results = [self.entity_infos[idx] for idx, _ in all_scores[:10]]  # Top 10 natijalarni olish
            else:
                # Agar embeddinglar keshi mavjud bo'lsa, to'g'ridan-to'g'ri hisoblash
                cos_scores = util.cos_sim(query_embedding, self.entity_embeddings_cache)[0]
                
                # Natijalarni saralash
                cos_scores_tensor = torch.tensor(cos_scores)
                sorted_indices = torch.argsort(cos_scores_tensor, descending=True)
                
                # Top natijalarni olish
                results = [self.entity_infos[idx] for idx in sorted_indices[:10]]  # Top 10 natijalarni olish
            
            if return_raw:
                return results
            
            # Natijalarni formatlash
            formatted_results = []
            for i, result in enumerate(results[:5]):  # Top 5 natijalarni ko'rsatish
                formatted_results.append(f"{i+1}. Kod: {result['CODE']}, Nomi: {result['NAME']}")
            
            formatted_text = "\n".join(formatted_results)
            
            if len(results) > 5:
                formatted_text += f"\n\nJami {len(results)} ta natija topildi."
            
            return formatted_text
        except Exception as e:
            logging.error(f"Semantik qidiruvda xatolik: {str(e)}")
            return [] if return_raw else f"Semantik qidiruvda xatolik: {str(e)}"

    def _search_by_code(self, code: str) -> Optional[Dict]:
        """Search for a specific code in the nation data."""
        for entity in self.nation_data:
            if entity['CODE'] == code:
                return entity
        return None

    def _search_by_name(self, name: str) -> Optional[Dict]:
        """Search for a specific name in the nation data."""
        for entity in self.nation_data:
            if entity['NAME'].lower() == name.lower():
                return entity
        return None

    def _get_general_info(self) -> str:
        """Get general information about the nation data."""
        return """
        Nation ma'lumotlari:
        
        """

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
        """Search for a specific text in the nation data."""

        # Muhim so'zlarni ajratib olish
        important_words = self._extract_important_words(query)
        logging.info(f"Matn qidiruv uchun muhim so'zlar: {important_words}")
        
        # Natijalarni saqlash uchun ro'yxat
        results = []

        for entity in self.nation_data:
            if entity['NAME'].lower() == query.lower() or entity['CODE'] == query:
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

    def _search_by_name(self, name: str) -> Optional[Dict]:
        """Search for a specific name in the nation data."""
        for entity in self.nation_data:
            if entity['NAME'].lower() == name.lower():
                return entity
        return None

    def _search_by_code(self, code: str) -> Optional[Dict]:
        """Search for a specific code in the nation data."""
        for entity in self.nation_data:
            if entity['CODE'].lower() == code.lower():
                return entity
        return None

    def _format_results(self, results):
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

    def _search_nation_data(self, query: str) -> Union[List[Dict], str]:
        """Search for a specific query in the nation data."""
        try:
            if not self.nation_data:
                return "Nation data yuklanmagan."
            
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

            # Natijalarni qaytarish
            if all_results:
                logging.info(f"Hibrid qidiruv muvaffaqiyatli natija berdi: {query}")
                return all_results
            
            # Natija topilmagan bo'lsa, bo'sh ro'yxat qaytarish
            return []
        except Exception as e:
            logging.error(f"Nation data qidirishda xatolik: {str(e)}")
            return f"Qidirishda xatolik: {str(e)}"

    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # So'rovni qayta ishlash va Nation ma'lumotlaridan tegishli ma'lumotni topish
            result = self._search_nation_data(query)
            
            # Agar natija bo'sh bo'lsa yoki xatolik haqida xabar bo'lsa
            if not result:
                return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Agar natija string bo'lsa (xatolik xabari)
            if isinstance(result, str):
                return f"Millat klassifikatori ma'lumotlari:\n\n{result}"
            
            # Agar natija ro'yxat bo'lsa (lug'atlar ro'yxati)
            if isinstance(result, list):
                formatted_results = self._format_results(result)
                return f"Millat klassifikatori ma'lumotlari:\n\n{formatted_results}"
            
            # Boshqa holatlar uchun
            return f"Millat klassifikatori ma'lumotlari:\n\n{str(result)}"
            
        except Exception as e:
            logging.error(f"Nation qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"