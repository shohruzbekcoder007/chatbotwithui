import json
import os
import re
import torch
import uuid
import logging
from typing import Dict, List, Optional, Union, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import util
from retriever.langchain_chroma import CustomEmbeddingFunction

# CustomEmbeddingFunction klassi retriever.langchain_chroma modulidan import qilingan

class DBIBTToolInput(BaseModel):
    query: str = Field(description="DBIBT ma'lumotlarini qidirish uchun so'rov. Bu yerga DBIBT kodi (masalan, '08824'), tashkilot nomi (masalan, 'O'zbekiston Respublikasi Vazirlar Mahkamasi'), OKPO/KTUT yoki STIR/INN raqami kiritilishi mumkin. STIR va INN bir xil ma'noda, OKPO va KTUT ham bir xil ma'noda ishlatiladi. Agar aniq DBIBT kodi ma'lum bo'lsa, faqat raqamni kiriting (masalan, '07474').")

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DBIBTTool(BaseTool):
    """DBIBT ma'lumotlarini qidirish uchun tool."""
    name: str = "dbibt_tool"
    description: str = "O'zbekiston Respublikasi Davlat va xo'jalik boshqaruvi idoralarini belgilash tizimi (DBIBT) ma'lumotlarini qidirish uchun tool. Bu tool orqali DBIBT kodi, tashkilot nomi, OKPO yoki INN raqami bo'yicha qidiruv qilish mumkin. Masalan: '08824', 'Vazirlar Mahkamasi', yoki '07474' kabi so'rovlar bilan qidiruv qilish mumkin."
    args_schema: Type[BaseModel] = DBIBTToolInput
    
    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    dbibt_data: Dict = Field(default_factory=dict)
    entity_DBIBT: List[str] = Field(default_factory=list)
    entity_OKPO: List[str] = Field(default_factory=list)
    entity_INN: List[str] = Field(default_factory=list)
    entity_NAME: List[str] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    external_embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)
    
    def __init__(self, dbibt_file_path: str, use_embeddings: bool = True, embedding_model=None):
        """Initialize the DBIBT tool with the path to the DBIBT JSON file."""
        super().__init__()
        self.dbibt_data = self._load_dbibt_data(dbibt_file_path)
        self.use_embeddings = use_embeddings
        self.external_embedding_model = embedding_model
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            self._prepare_embedding_data()
        else:
            # Lazy loading - embedding modeli kerak bo'lganda yuklanadi
            self.embedding_model = None

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
        
        # Embedding keshini tozalash
        self.entity_embeddings_cache = None

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

    def _search_by_text(self, query: str, return_raw: bool = False) -> Union[List[Dict], str]:
        """Search for a specific text in the DBIBT data."""
        try:
            # Muhim so'zlarni ajratib olish
            important_words = self._extract_important_words(query)
            logging.info(f"Matn qidiruv uchun muhim so'zlar: {important_words}")
            
            # STIR va INN bir xil ma'noda ekanligini tekshirish
            query_lower = query.lower()
            if "stir" in query_lower:
                query = query_lower.replace("stir", "").strip()
                logging.info(f"STIR so'rovi INN sifatida qidirilmoqda: {query}")
            
            # OKPO va KTUT bir xil ma'noda ekanligini tekshirish
            if "ktut" in query_lower:
                query = query_lower.replace("ktut", "").strip()
                logging.info(f"KTUT so'rovi OKPO sifatida qidirilmoqda: {query}")
            
            # Natijalarni saqlash uchun ro'yxat
            results = []

            # Exact match qidirish
            for entity in self.dbibt_data:
                try:
                    # DBIBT kodi bo'yicha qidirish (eng muhim)
                    if 'DBIBT' in entity and entity['DBIBT'] == query:
                        results.append(entity)
                        logging.info(f"DBIBT kodi bo'yicha topildi: {entity['DBIBT']}")
                        
                    # Boshqa maydonlar bo'yicha qidirish
                    elif ('NAME' in entity and entity['NAME'].lower() == query.lower() or
                          'OKPO' in entity and entity['OKPO'] == query or
                          'INN' in entity and entity['INN'] == query):
                        results.append(entity)
                except (TypeError, AttributeError, KeyError) as e:
                    logging.warning(f"Entity processing error: {str(e)}, Entity: {type(entity)}")
                    continue

            # Agar exact match topilmasa, partial match qidirish
            if not results:
                for entity in self.dbibt_data:
                    try:
                        if isinstance(entity, dict) and 'NAME' in entity:
                            entity_name = entity['NAME'].lower()
                            # Muhim so'zlar bo'yicha qidirish
                            for word in important_words:
                                if word.lower() in entity_name:
                                    if entity not in results:
                                        results.append(entity)
                                    break
                    except (TypeError, AttributeError, KeyError) as e:
                        logging.warning(f"Entity name processing error: {str(e)}")
                        continue

            # Debug uchun natijalarni ko'rsatish
            if results:
                logging.info(f"Topilgan natijalar soni: {len(results)}")
                if len(results) > 0:
                    logging.info(f"Birinchi natija: {results[0]}")
            else:
                logging.info("Natijalar topilmadi")

            # Natijalarni qaytarish
            if return_raw:
                return results
            
            if results:
                formatted_results = []
                for i, result in enumerate(results[:5]):  # Top 5 natijalarni ko'rsatish
                    try:
                        if isinstance(result, dict):
                            formatted_results.append(
                                f"{i+1}. Kod: {result.get('DBIBT', 'N/A')}, "
                                f"OKPO/KTUT: {result.get('OKPO', 'N/A')}, "  # OKPO ni OKPO/KTUT sifatida ko'rsatish
                                f"STIR/INN: {result.get('INN', 'N/A')}, "
                                f"Nomi: {result.get('NAME', 'N/A')}"
                            )
                    except Exception as e:
                        logging.error(f"Result formatting error: {str(e)}")
                
                formatted_text = "\n".join(formatted_results)
                
                if len(results) > 5:
                    formatted_text += f"\n\nJami {len(results)} ta natija topildi."
                
                return formatted_text
            
            return "Ma'lumot topilmadi."
            
        except Exception as e:
            logging.error(f"Text search error: {str(e)}")
            return f"Qidirishda xatolik: {str(e)}"

    def _semantic_search(self, query: str, return_raw: bool = False) -> Union[List[Dict], str]:
        """Semantik qidiruv - embedding modelini ishlatib o'xshash ma'lumotlarni topish"""
        try: 
            if not self.entity_DBIBT or not self.entity_OKPO or not self.entity_INN or not self.entity_NAME:
                return [] if return_raw else "Ma'lumotlar mavjud emas."
            
            if not self.use_embeddings:
                return [] if return_raw else "Semantik qidiruv o'chirilgan."

            # STIR va INN bir xil ma'noda ekanligini tekshirish
            query_lower = query.lower()
            if "stir" in query_lower:
                query = query_lower.replace("stir", "").strip()
                logging.info(f"Semantik qidiruvda STIR so'rovi INN sifatida qidirilmoqda: {query}")
            
            # OKPO va KTUT bir xil ma'noda ekanligini tekshirish
            if "ktut" in query_lower:
                query = query_lower.replace("ktut", "").strip()
                logging.info(f"Semantik qidiruvda KTUT so'rovi OKPO sifatida qidirilmoqda: {query}")

            # Embedding modelini kerak bo'lganda yuklash (lazy loading)
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
                self._prepare_embedding_data()
                
            model = self.embedding_model.model
            batch_size = 100  # Bir vaqtda qayta ishlanadigan hujjatlar soni
            top_k = 5  # Eng yuqori o'xshashlikdagi natijalar soni

            # So'rov embeddingini olish
            query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            
            # Entity ma'lumotlarini tayyorlash
            entity_texts = [f"{entity['DBIBT']} {entity['NAME']} {entity['OKPO']} {entity['INN']}" for entity in self.dbibt_data]
            
            # Barcha ma'lumotlarni embedding qilish
            # Agar entity_embeddings_cache mavjud bo'lmasa, yangi embeddinglarni hisoblash
            if not hasattr(self, 'entity_embeddings_cache') or self.entity_embeddings_cache is None:
                logging.info("DBIBT embeddinglar keshi yaratilmoqda...")
                self.entity_embeddings_cache = torch.zeros((len(entity_texts), model.get_sentence_embedding_dimension()))
                
                # Hujjatlarni batch usulida qayta ishlash
                for i in range(0, len(entity_texts), batch_size):
                    batch_texts = entity_texts[i:i+batch_size]
                    
                    # Batch embeddinglarini olish
                    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    
                    # Embeddinglarni keshga saqlash
                    self.entity_embeddings_cache[i:i+len(batch_texts)] = batch_embeddings
            
            # Eng o'xshash ma'lumotlarni topish
            cos_scores = util.cos_sim(query_embedding, self.entity_embeddings_cache)[0]
            
            # Top natijalarni olish
            top_k = min(top_k, len(cos_scores))
            if top_k == 0:
                return [] if return_raw else ""
                
            top_results = torch.topk(cos_scores, k=top_k)
            
            # Natijalarni formatlash
            results = []
            for score, idx in zip(top_results[0], top_results[1]):
                if score > 0.5:  # Minimal o'xshashlik chegarasi
                    results.append((score.item(), self.dbibt_data[idx]))
            
            # Natijalarni o'xshashlik darajasi bo'yicha saralash
            results.sort(key=lambda x: x[0], reverse=True)
            
            if return_raw:
                return [item for _, item in results]
            
            # Natijalarni formatlash
            formatted_results = []
            for i, (score, result) in enumerate(results):
                formatted_results.append(
                    f"{i+1}. Kod: {result['DBIBT']}, OKPO/KTUT: {result['OKPO']}, "
                    f"STIR/INN: {result['INN']}, Nomi: {result['NAME']} "
                    f"(o'xshashlik: {score:.2f})"
                )
            
            formatted_text = "\n".join(formatted_results)
            
            if len(results) > 0:
                return formatted_text
            return ""
                
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

            # STIR va INN bir xil ma'noda ekanligini tekshirish
            if "stir" in query:
                query = query.replace("stir", "").strip()
                logging.info(f"STIR so'rovi INN sifatida qidirilmoqda: {query}")
            
            # OKPO va KTUT bir xil ma'noda ekanligini tekshirish
            if "ktut" in query:
                query = query.replace("ktut", "").strip()
                logging.info(f"KTUT so'rovi OKPO sifatida qidirilmoqda: {query}")

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
                formatted_results.append(f"{i+1}. Kod: {result['DBIBT']}, OKPO/KTUT: {result['OKPO']}, STIR/INN: {result['INN']}, Nomi: {result['NAME']}")
            
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
    
    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # STIR va INN bir xil ma'noda ekanligini tekshirish
            query_lower = query.lower()
            if "stir" in query_lower:
                query = query_lower.replace("stir", "").strip()
                logging.info(f"STIR so'rovi INN sifatida qidirilmoqda: {query}")
            
            # OKPO va KTUT bir xil ma'noda ekanligini tekshirish
            if "ktut" in query_lower:
                query = query_lower.replace("ktut", "").strip()
                logging.info(f"KTUT so'rovi OKPO sifatida qidirilmoqda: {query}")
            
            # DBIBT kodi bo'yicha to'g'ridan-to'g'ri qidirish
            if query.isdigit() or (len(query) == 5 and query[:5].isdigit()):
                # Agar so'rov faqat raqamlardan iborat bo'lsa, DBIBT kodi sifatida qabul qilamiz
                for entity in self.dbibt_data:
                    if 'DBIBT' in entity and entity['DBIBT'] == query:
                        return f"DBIBT ma'lumotlari:\n\n1. Kod: {entity['DBIBT']}, OKPO/KTUT: {entity['OKPO']}, STIR/INN: {entity['INN']}, Nomi: {entity['NAME']}"
            
            # Agar to'g'ridan-to'g'ri topilmasa, oddiy qidiruv
            result = self._search_dbibt_data(query)
            
            # Agar natija bo'sh bo'lsa yoki xatolik haqida xabar bo'lsa
            if not result:
                return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Agar natija string bo'lsa (xatolik xabari)
            if isinstance(result, str):
                return f"DBIBT ma'lumotlari:\n\n{result}"
            
            # Agar natija ro'yxat bo'lsa (lug'atlar ro'yxati)
            if isinstance(result, list):
                formatted_results = []
                for i, item in enumerate(result[:5]):  # Top 5 natijalarni ko'rsatish
                    try:
                        if isinstance(item, dict):
                            formatted_results.append(
                                f"{i+1}. Kod: {item.get('DBIBT', 'N/A')}, "
                                f"OKPO/KTUT: {item.get('OKPO', 'N/A')}, "
                                f"STIR/INN: {item.get('INN', 'N/A')}, "
                                f"Nomi: {item.get('NAME', 'N/A')}"
                            )
                    except Exception as e:
                        logging.error(f"Result formatting error: {str(e)}")
                
                formatted_text = "\n".join(formatted_results)
                
                # Agar natijalar ko'p bo'lsa, qo'shimcha ma'lumot
                if len(result) > 5:
                    formatted_text += f"\n\nJami {len(result)} ta natija topildi."
                
                return f"DBIBT ma'lumotlari:\n\n{formatted_text}"
            
            # Boshqa holatlar uchun
            return f"DBIBT ma'lumotlari:\n\n{str(result)}"
            
        except Exception as e:
            logging.error(f"DBIBT qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"