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

class THSHTool(BaseTool):
    """Tashkiliy-huquqiy shakllar ma'lumotlarini qidirish uchun tool."""
    name: str = "thsh_tool"
    description: str = "Tashkiliy-huquqiy shakllar (THSH) ma'lumotlarini qidirish uchun mo'ljallangan vosita. Bu tool orqali tashkiliy-huquqiy shakllarning kodi, nomi, qisqa nomi va boshqa ma'lumotlar bo'yicha qidiruv qilish mumkin. Misol uchun: '110' (Xususiy korxona), 'Aksiyadorlik jamiyati', 'MJ' (Mas'uliyati cheklangan jamiyat) kabi so'rovlar orqali ma'lumotlarni izlash mumkin."
    
    thsh_data: List[Dict] = Field(default_factory=list)
    use_embeddings: bool = Field(default=False)
    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)
    
    def __init__(self, thsh_file_path: str, use_embeddings: bool = True, embedding_model=None):
        """Initialize the THSH tool with the path to the THSH JSON file."""
        super().__init__()
        self.thsh_data = self._load_thsh_data(thsh_file_path)
        self.use_embeddings = use_embeddings
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logging.info("Tashqaridan berilgan embedding modeli o'rnatildi.")
        else:
            # Embedding modelini kerak bo'lganda yuklaymiz
            self.embedding_model = None
    
    def _load_thsh_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load THSH data from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"THSH data file not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # So'rovni qayta ishlash va THSH ma'lumotlaridan tegishli ma'lumotni topish
            result = self._search_thsh_data(query)
            if not result:
                return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Agar natijada "Tashkiliy-huquqiy shakl ma'lumotlari" qismi bo'lmasa, qo'shish
            if not result.startswith("Tashkiliy-huquqiy shakl ma'lumotlari"):
                result = f"Tashkiliy-huquqiy shakl ma'lumotlari:\n\n{result}"
            
            # Natijani aniq formatda qaytarish
            return result
        except Exception as e:
            logging.error(f"THSH ma'lumotlarini qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"
    
    def _prepare_embedding_data(self):
        """Embedding uchun ma'lumotlarni tayyorlash"""
        self.entity_texts = []
        self.entity_infos = []
        
        # THSH ma'lumotlarini qo'shish
        for thsh in self.thsh_data:
            thsh_text = f"{thsh.get('CODE', '')} {thsh.get('name', '')} {thsh.get('shortname', '')} {thsh.get('groupcode', '')}"
            self.entity_texts.append(thsh_text)
            self.entity_infos.append(("thsh", thsh))
        
        # Embeddinglarni kerak bo'lganda hisoblaymiz
        try:
            if self.use_embeddings and self.entity_texts:
                logging.info(f"THSH uchun {len(self.entity_texts)} ta embedding ma'lumoti tayyorlandi.")
                # Embeddinglarni kerak bo'lganda hisoblaymiz
                self.entity_embeddings_cache = None
        except Exception as e:
            logging.error(f"THSH embeddinglarini tayyorlashda xatolik: {str(e)}")
    
    def _search_thsh_data(self, query: str) -> str:
        """THSH ma'lumotlaridan so'rov bo'yicha ma'lumot qidirish."""
        if not self.thsh_data:
            return "THSH ma'lumotlari yuklanmagan."
        
        # Agar so'rov bo'sh bo'lsa
        if not query.strip():
            return "Iltimos, qidiruv uchun so'rov kiriting."
        
        # So'rovni katta harflarga o'tkazish
        query_upper = query.upper()
        
        # Agar so'rov "ro'yxat" yoki "list" so'zlarini o'z ichiga olsa, ro'yxatni qaytarish
        # Lekin "guruhi" so'zi bo'lsa, bu qismni o'tkazib yuboramiz
        if re.search(r'ro[\'\']yxat|list|barcha|hammasi', query.lower()) and not ("GURUHI" in query_upper or "GURUH" in query_upper):
            return self._get_thsh_list()
        
        # Semantik qidiruv
        if self.use_embeddings and self.embedding_model is not None:
            try:
                semantic_result = self._semantic_search(query)
                if semantic_result:
                    return semantic_result
            except Exception as e:
                logging.error(f"Semantik qidirishda xatolik: {str(e)}")
        
        # Oddiy qidiruv (agar semantik qidiruv natija bermasa)
        matching_thsh = []
        
        # Guruh kodi bo'yicha qidiruv ("thsh guruhi 200" kabi so'rovlar uchun)
        if "GURUHI" in query_upper or "GURUH" in query_upper:
            # So'rovdan raqamni ajratib olish
            numbers = re.findall(r'\d+', query)
            if numbers:
                group_code = numbers[0]
                # Barcha shu guruh kodiga mos keladigan THSH larni topish
                for thsh in self.thsh_data:
                    if thsh.get('groupcode', '') == group_code:
                        matching_thsh.append(thsh)
                if matching_thsh:
                    result = "Tashkiliy-huquqiy shakl ma'lumotlari:\n\n"
                    for thsh in matching_thsh:
                        result += self._format_thsh_info(thsh) + "\n\n"
                    return result
        
        # Kod, guruh kodi, nom va qisqa nom bo'yicha qidiruv
        for thsh in self.thsh_data:
            code = thsh.get('CODE', '').upper()
            groupcode = thsh.get('groupcode', '').upper()
            name = thsh.get('name', '').upper()
            shortname = thsh.get('shortname', '').upper()
            
            # Guruh kodi bo'yicha qidiruv
            if "GURUH KODI" in query_upper and groupcode in query_upper:
                matching_thsh.append(thsh)
            # Guruh kodi raqami bo'yicha qidiruv
            elif "GURUH" in query_upper and groupcode == query_upper.split()[-1]:
                matching_thsh.append(thsh)
            # Oddiy qidiruv
            elif query_upper == code or query_upper in name or query_upper == shortname or query_upper == groupcode:
                matching_thsh.append(thsh)
        
        # Agar kod bo'yicha topilmasa, nom bo'yicha qidiruv
        if not matching_thsh:
            for thsh in self.thsh_data:
                name = thsh.get('name', '').upper()
                if query_upper in name:
                    matching_thsh.append(thsh)
        
        # Natijalarni formatlash
        if matching_thsh:
            if len(matching_thsh) == 1:
                return self._format_thsh_info(matching_thsh[0])
            else:
                result = "Tashkiliy-huquqiy shakl ma'lumotlari:\n\n"
                for thsh in matching_thsh[:20]:  # Ko'proq natijalarni ko'rsatish (5 o'rniga 20)
                    result += self._format_thsh_info(thsh) + "\n\n"
                
                if len(matching_thsh) > 20:
                    result += f"...va yana {len(matching_thsh) - 20} ta natija mavjud."
                
                return result
        
        return "So'rov bo'yicha ma'lumot topilmadi."
    
    def _semantic_search(self, query: str) -> str:
        """Semantik qidirish orqali THSH ma'lumotlarini topish"""
        try:
            if not self.use_embeddings:
                return "Semantik qidiruv o'chirilgan."
                
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
                
            model = self.embedding_model.model
            
            # Device ni aniqlash (CPU yoki CUDA)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Xotira tejash uchun batch usulida ishlash
            batch_size = 100  # Bir vaqtda qayta ishlanadigan hujjatlar soni
            top_k = 3  # Eng yuqori o'xshashlikdagi natijalar soni
            
            # So'rov embeddingini olish va device ga joylashtirish
            query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            if hasattr(query_embedding, 'to'):
                query_embedding = query_embedding.to(device)
            
            # Barcha ma'lumotlarni embedding qilish
            # Agar entity_embeddings_cache mavjud bo'lmasa, yangi embeddinglarni hisoblash
            if not hasattr(self, 'entity_embeddings_cache') or self.entity_embeddings_cache is None:
                logging.info("Embeddinglar keshi yaratilmoqda...")
                
                # Natijalarni saqlash uchun
                all_scores = []
                
                # Hujjatlarni batch usulida qayta ishlash
                for i in range(0, len(self.entity_texts), batch_size):
                    batch_texts = self.entity_texts[i:i+batch_size]
                    
                    # Batch embeddinglarini olish va device ga joylashtirish
                    batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    if hasattr(batch_embeddings, 'to'):
                        batch_embeddings = batch_embeddings.to(device)
                    
                    # Har bir embedding uchun o'xshashlikni hisoblash
                    for j, emb in enumerate(batch_embeddings):
                        # Embeddingni to'g'ri device ga ko'chirish
                        if hasattr(emb, 'to'):
                            emb = emb.to(device)
                        
                        # Cosine similarity hisoblash - device consistency bilan
                        cos_score = util.cos_sim(query_embedding.unsqueeze(0), emb.unsqueeze(0))[0][0].item()
                        all_scores.append((i+j, cos_score))
                
                # O'xshashlik bo'yicha saralash
                all_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Top natijalarni olish
                results = []
                for idx, score in all_scores[:top_k]:
                    if score > 0.5 and idx < len(self.entity_infos):  # Minimal o'xshashlik chegarasi
                        entity_type, entity_info = self.entity_infos[idx]
                        if entity_type == "thsh":
                            results.append((score, self._format_thsh_info(entity_info)))
            else:
                # Agar mavjud cache boshqa device da bo'lsa, uni to'g'ri device ga ko'chirish
                if hasattr(self.entity_embeddings_cache, 'device') and self.entity_embeddings_cache.device != device:
                    self.entity_embeddings_cache = self.entity_embeddings_cache.to(device)
                
                # Eng o'xshash ma'lumotlarni topish
                cos_scores = util.cos_sim(query_embedding, self.entity_embeddings_cache)[0]
                
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
                        if entity_type == "thsh":
                            results.append((score.item(), self._format_thsh_info(entity_info)))
            
            # Natijalarni o'xshashlik darajasi bo'yicha saralash
            results.sort(key=lambda x: x[0], reverse=True)
            
            if results:
                if len(results) == 1:
                    return results[0][1]
                else:
                    result = "Tashkiliy-huquqiy shakl ma'lumotlari:\n\n"
                    for _, thsh_info in results:
                        result += thsh_info + "\n\n"
                    return result
            return ""
        except Exception as e:
            logging.error(f"Semantik qidirishda xatolik: {str(e)}")
            # Xatolik bo'lganda oddiy qidiruv natijasini qaytarish
            matching_thsh = []
            query_upper = query.upper()
            for thsh in self.thsh_data:
                name = thsh.get('name', '').upper()
                code = thsh.get('CODE', '').upper()
                groupcode = thsh.get('groupcode', '').upper()
                
                if query_upper in name or query_upper == code or query_upper == groupcode:
                    matching_thsh.append(thsh)
            
            if matching_thsh:
                if len(matching_thsh) == 1:
                    return self._format_thsh_info(matching_thsh[0])
                else:
                    result = "Tashkiliy-huquqiy shakl ma'lumotlari:\n\n"
                    for thsh in matching_thsh[:3]:  # Faqat birinchi 3 tasini ko'rsatish
                        result += self._format_thsh_info(thsh) + "\n\n"
                    return result
        
        return ""
    
    def _format_thsh_info(self, thsh: Dict) -> str:
        """THSH ma'lumotlarini formatlash"""
        info = f"Kod: {thsh.get('CODE', 'Mavjud emas')}\n"
        info += f"Guruh kodi: {thsh.get('groupcode', 'Mavjud emas')}\n"
        info += f"Nomi: {thsh.get('name', 'Mavjud emas')}\n"
        info += f"Qisqa nomi: {thsh.get('shortname', 'Mavjud emas')}\n"
        info += f"Muvofiqlik: {thsh.get('correspondence', 'Mavjud emas')}"
        
        return info
    
    def _get_thsh_list(self) -> str:
        """Barcha THSH ro'yxatini olish"""
        if not self.thsh_data:
            return "THSH ro'yxati topilmadi."
        
        result = "Tashkiliy-huquqiy shakllar ro'yxati:\n\n"
        for i, thsh in enumerate(self.thsh_data[:20], 1):  # Faqat birinchi 20 ta THSH ni ko'rsatish
            result += f"{i}. {thsh.get('name', 'Mavjud emas')} ({thsh.get('CODE', 'Mavjud emas')})\n"
        
        if len(self.thsh_data) > 20:
            result += f"\n...va yana {len(self.thsh_data) - 20} ta THSH mavjud."
        
        return result