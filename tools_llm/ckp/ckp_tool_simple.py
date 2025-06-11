import json
import os
import re
import logging
import torch
from typing import Dict, List, Any, Optional, Union, Type
from sentence_transformers import util
from retriever.langchain_chroma import CustomEmbeddingFunction
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CkpToolInput(BaseModel):
    query: str = Field(description="MST (Mahsulotlarning tasniflagichi) ma'lumotlarini qidirish uchun so'rov")


class CkpTool(BaseTool):
    """MST (Mahsulotlarning tasniflagichi) ma'lumotlarini qidirish uchun tool"""
    name: str = "ckp_tool"
    description: str = "MST (Mahsulotlarning tasniflagichi) ma\"lumotlarini qidirish va tahlil qilish uchun mo\"ljallangan vosita. Bu tool orqali mahsulotlar kodlari, nomlari va tasniflarini izlash, ularning ma\"lumotlarini ko\"rish va tahlil qilish mumkin. Qidiruv so\"z, kod yoki tasnif bo\"yicha amalga oshirilishi mumkin."
    args_schema: Type[BaseModel] = CkpToolInput
    
    # Pydantic v2 uchun barcha maydonlarni oldindan e'lon qilish
    ckp_data: List[Dict] = Field(default_factory=list)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    embedding_model: Optional[Any] = Field(default=None)
    use_embeddings: bool = Field(default=True)
    external_embedding_model: Optional[Any] = Field(default=None)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)
    
    def __init__(self, ckp_file_path: str, use_embeddings: bool = True, embedding_model=None):
        super().__init__()
        self.use_embeddings = use_embeddings
        self.external_embedding_model = embedding_model
        
        # Ma'lumotlarni yuklash
        self._load_data(ckp_file_path)
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            self._prepare_embedding_data()
        else:
            # Lazy loading - embedding modeli kerak bo'lganda yuklanadi
            self.embedding_model = None
    
    def _load_data(self, ckp_file_path: str):
        """CKP ma'lumotlarini yuklash"""
        try:
            # Ma'lumotlar fayli
            if not ckp_file_path:
                raise ValueError("CKP ma'lumotlari fayli ko'rsatilmagan")
                
            # Absolute path bo'lmasa, relative path sifatida ko'rib chiqish
            if os.path.isabs(ckp_file_path):
                data_file = ckp_file_path
            else:
                # Relative path bo'lsa, current working directory dan olish
                # Agar path tools_llm/ bilan boshlansa, uni olib tashlash kerak
                if ckp_file_path.startswith('tools_llm/'):
                    # tools_llm/ prefixini olib tashlash
                    relative_path = ckp_file_path[len('tools_llm/'):]
                    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', relative_path)
                else:
                    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckp_file_path)
                
                # Path ni normalize qilish
                data_file = os.path.normpath(data_file)
            
            logging.info(f"CKP ma'lumotlari fayli: {data_file}")
            
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"CKP ma'lumotlari fayli topilmadi: {data_file}")
            
            # Ma'lumotlarni yuklash
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            # Ma'lumotlarni to'g'ri formatga o'tkazish
            self.ckp_data = []
            for item in raw_data:
                code = item.get("CODE", "")
                name = item.get("NAME", "")
                # Har bir elementni content maydoniga ega bo'lgan lug'at ko'rinishiga o'tkazish
                content = f"Kod: {code}\nNomi: {name}"
                self.ckp_data.append({"content": content})
                
            logging.info(f"CKP ma'lumotlari yuklandi: {len(self.ckp_data)} ta element")
        except Exception as e:
            logging.error(f"CKP ma'lumotlarini yuklashda xatolik: {str(e)}")
            self.ckp_data = []
            raise RuntimeError(f"CKP ma'lumotlarini yuklashda xatolik: {str(e)}")
    
    def _load_embedding_model(self):
        """Embedding modelini yuklash (lazy loading)"""
        try:
            if not self.use_embeddings:
                logging.info("Embedding o'chirilgan")
                return
                
            # Agar model tashqaridan berilgan bo'lsa, uni ishlatish
            if self.external_embedding_model is not None:
                self.embedding_model = self.external_embedding_model
                logging.info("Tashqi embedding modeli ishlatilmoqda")
                return
                
            # Agar model yuklanmagan bo'lsa, yangi model yaratish
            if self.embedding_model is None:
                try:
                    # Embedding modelini yuklash
                    logging.info("Embedding modeli yuklanmoqda...")
                    self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
                    logging.info("Embedding modeli yuklandi")
                except Exception as e:
                    logging.error(f"Embedding modelini yuklashda xatolik: {str(e)}")
                    self.embedding_model = None
                    self.use_embeddings = False
                
        except Exception as e:
            logging.error(f"Embedding modelini yuklashda xatolik: {str(e)}")
            self.embedding_model = None
    
    def _prepare_embedding_data(self):
        """Embedding uchun ma'lumotlarni tayyorlash"""
        try:
            # Entity ma'lumotlarini tayyorlash
            self.entity_texts = [item.get('content', '') for item in self.ckp_data]
            self.entity_infos = self.ckp_data
            
            # Embedding keshini tozalash
            self.entity_embeddings_cache = None
        except Exception as e:
            logging.error(f"Embedding ma'lumotlarini tayyorlashda xatolik: {str(e)}")
    
    def _run(self, query: str) -> str:
        """Tool ishga tushirilganda bajariladigan asosiy metod
        
        Args:
            query: Foydalanuvchi so'rovi
            
        Returns:
            Formatlangan natija matni
        """
        # Ma'lumotlar yuklanganligini tekshirish
        if not self.ckp_data:
            return "CKP ma'lumotlari yuklanmagan. Iltimos, ma'lumotlar faylini tekshiring."
            
        # So'rovni tozalash
        query = query.strip()
        
        # Agar so'rov bo'sh bo'lsa
        if not query:
            return self._get_general_info()
        
        return self._search_ckp_data(query)
    
    def _search_ckp_data(self, query: str) -> str:
        """CKP ma'lumotlaridan so'rov bo'yicha ma'lumot qidirish
        
        Args:
            query: Qidiruv so'rovi
            
        Returns:
            Formatlangan natija matni
        """
        try:
            if not self.ckp_data:
                return "CKP ma'lumotlari yuklanmagan."
            
            # So'rovni tozalash
            original_query = query
            query = query.lower()
            # Nuqtalarni saqlab qolish, chunki ular CKP kodlari uchun muhim
            query = re.sub(r'[^\w\s\.]', '', query)
            logging.info(f"Qidirilayotgan so'rov: '{original_query}', tozalangan so'rov: '{query}'")
            
            # 1. Kod qidirish - kod topilsa va natija bo'lsa, uni qaytarish
            # Kod formati: raqamlar va nuqtalardan iborat (masalan 01.11.12.1)
            code_match = re.search(r'\b(\d+(?:\.\d+){0,3})\b', original_query)
            if code_match:
                code = code_match.group(1)
                logging.info(f"So'rovdan kod topildi: {code}")
                code_result = self._search_by_code(code)
                if code_result:
                    return code_result
                else:
                    # Kod topildi, lekin natija yo'q
                    logging.info(f"Kod {code} uchun natija topilmadi. Prefiks qidiruv va semantik qidiruvga o'tilmoqda.")
                    
                    # Kod prefiksini olish (masalan 01.11.31.9 dan 01.11.31 ni olish)
                    code_parts = code.split('.')
                    if len(code_parts) > 1:
                        # Oxirgi qismni olib tashlash
                        prefix_code = '.'.join(code_parts[:-1])
                        logging.info(f"Prefiks kod bo'yicha qidirilmoqda: {prefix_code}")
                        prefix_result = self._search_by_code(prefix_code)
                        
                        if prefix_result:
                            # Prefiks kod bo'yicha natija topildi
                            if "qaysi mahsulot" in original_query.lower():
                                # Prefiks natijasidan MST ma'lumotlari: qismini olib tashlash
                                clean_result = prefix_result.replace("MST ma'lumotlari:", "")
                                return f"MST ma'lumotlari:\n\n{code} kodi uchun aniq ma'lumot topilmadi, lekin {prefix_code} kodi quyidagi mahsulotga tegishli:\n\n{clean_result}"
                            return prefix_result
                    
                    # Agar kod aniq ko'rsatilgan bo'lsa, lekin natija topilmasa
                    if "qaysi mahsulot" in original_query.lower() or "qaysi kod" in original_query.lower():
                        return f"MST ma'lumotlari:\n\n{code} kodi bo'yicha ma'lumot topilmadi. Iltimos, kodni tekshiring yoki boshqa so'rov bilan qayta urinib ko'ring."
            
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
                    if word in result.lower():
                        score += 1
                return score
            
            # Natijalarni saralash
            all_results.sort(key=score_result, reverse=True)
            
            # Natijalarni formatlash va qaytarish
            if all_results:
                logging.info(f"Hibrid qidiruv muvaffaqiyatli natija berdi: {query}")
                return self._format_results(all_results[:5])
            
            # Natija topilmadi
            return "MST ma'lumotlari bo'yicha natija topilmadi."
            
        except Exception as e:
            logging.error(f"CKP qidirishda xatolik: {str(e)}")
            return f"MST ma'lumotlari:\n\nQidirishda xatolik yuz berdi: {str(e)}"

    def _get_general_info(self) -> str:
        """MST haqida umumiy ma'lumot olish"""
        # Birinchi elementni olish - bu umumiy ma'lumot
        if self.ckp_data and len(self.ckp_data) > 0:
            general_info = self.ckp_data[0].get("content", "")
            return general_info
        
        return "MST - bu O'zbekiston Respublikasi iqtisodiy faoliyat turlari bo'yicha mahsulotlarning (tovarlar, ishlar, xizmatlarning) statistik tasniflagichi"
    
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
    
    def _search_by_code(self, code: str) -> str:
        """Kod bo'yicha ma'lumot qidirish
        
        Args:
            code: Qidirilayotgan kod
            
        Returns:
            Formatlangan natija matni
        """
        logging.info(f"Kod bo'yicha qidirilmoqda: {code}")
        
        # Natijalar ro'yxati
        results = []
        
        # 1. Kod bo'yicha to'g'ridan-to'g'ri qidirish - bu eng aniq usul
        for item in self.ckp_data:
            item_code = item.get("CODE", "")
            if item_code == code:
                name = item.get("NAME", "")
                results.append(f"Kod: {item_code}\nNomi: {name}")
                logging.info(f"Aniq mos kelish topildi: {code}")
                # Aniq mos kelish topilganda, uni qaytaramiz
                return f"MST ma'lumotlari:\n\n{results[0]}"
        
        # 2. Kod prefiksi bo'yicha qidirish
        prefix_matches = []
        for item in self.ckp_data:
            item_code = item.get("CODE", "")
            if item_code.startswith(code):
                name = item.get("NAME", "")
                prefix_matches.append(f"Kod: {item_code}\nNomi: {name}")
                logging.info(f"Kod prefiksi bilan mos kelish topildi: {item_code}")
        
        # 3. Kod qismi bo'yicha qidirish
        if not prefix_matches:
            for item in self.ckp_data:
                item_code = item.get("CODE", "")
                if code in item_code:
                    name = item.get("NAME", "")
                    prefix_matches.append(f"Kod: {item_code}\nNomi: {name}")
                    logging.info(f"Kod qismi bilan mos kelindi: {item_code}")
        
        # Natijalarni birlashtirish
        results.extend(prefix_matches)
        
        # Natijalarni formatlash
        if results:
            if len(results) > 5:
                result_text = "\n\n".join(results[:5])
                result_text += f"\n\nJami {len(results)} ta natija topildi."
            else:
                result_text = "\n\n".join(results)
            
            return f"MST ma'lumotlari:\n\n{result_text}"
        
        # Natija topilmadi
        logging.info(f"Kod {code} uchun natija topilmadi.")
        return ""
    
    def _search_by_text(self, query: str, return_raw: bool = False) -> Union[str, List[str]]:
        """Matn bo'yicha qidirish
        
        Args:
            query: Qidirilayotgan so'rov
            return_raw: Natijalarni xom holda qaytarish
            
        Returns:
            Formatlangan natija matni yoki natijalar ro'yxati
        """
        # Muhim so'zlarni ajratib olish
        important_words = self._extract_important_words(query)
        logging.info(f"Matn qidiruv uchun muhim so'zlar: {important_words}")
        
        # Natijalarni saqlash uchun ro'yxat
        results = []
        
        # Matn bo'yicha qidirish - to'g'ridan-to'g'ri ma'lumotlar ro'yxatidan
        for item in self.ckp_data:
            content = item.get("content", "").lower()
            
            # Kod va nomni ajratib olish
            code_match = re.search(r'kod:\s*([^\n]+)', content, re.IGNORECASE)
            name_match = re.search(r'nomi:\s*([^\n]+)', content, re.IGNORECASE)
            
            item_code = code_match.group(1) if code_match else ""
            name = name_match.group(1).lower() if name_match else ""
            
            # Har bir muhim so'z uchun tekshirish
            match_count = 0
            for word in important_words:
                if word.lower() in content:
                    match_count += 1
            
            # Kamida bir so'z mos kelsa, natijaga qo'shish
            if match_count >= 1:
                results.append(content)
        
        # Natijalarni qaytarish
        if return_raw:
            return results
        
        if results:
            if len(results) > 5:
                result_text = "\n\n".join(results[:5])
                result_text += f"\n\nJami {len(results)} ta natija topildi."
            else:
                result_text = "\n\n".join(results)
            
            return f"MST ma'lumotlari:\n\n{result_text}"
        
        return ""

    def _semantic_search(self, query: str, return_raw: bool = False) -> Union[str, List[str]]:
        """Semantik qidiruv - embedding modelini ishlatib o'xshash ma'lumotlarni topish
        
        Args:
            query: Qidiruv so'rovi
            return_raw: Agar True bo'lsa, xom natijalar ro'yxatini qaytaradi
            
        Returns:
            Formatlangan matn yoki xom natijalar ro'yxati
        """
        if not self.entity_texts or not self.entity_infos:
            return [] if return_raw else "Embedding ma'lumotlari mavjud emas."
        
        if not self.use_embeddings:
            return [] if return_raw else "Semantik qidiruv o'chirilgan."
        
        try:
            # So'rovni tozalash va muhim so'zlarni ajratib olish
            clean_query = query.lower()
            
            # Muhim so'zlarni ajratib olish
            important_words = [word for word in clean_query.split() if len(word) > 2]
            
            # Embedding modelini kerak bo'lganda yuklash (lazy loading)
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                self._load_embedding_model()
                if self.embedding_model is None:
                    return [] if return_raw else "Embedding modeli yuklanmagan. Semantik qidiruv ishlamaydi."
            
            model = self.embedding_model.model
            batch_size = 100  # Bir vaqtda qayta ishlanadigan hujjatlar soni
            top_k = 5  # Eng yuqori o'xshashlikdagi natijalar soni
            
            # Device ni aniqlash (CPU yoki CUDA)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # So'rov embeddingini olish va device ga joylashtirish
            query_embedding = model.encode(clean_query, convert_to_tensor=True, show_progress_bar=False)
            if hasattr(query_embedding, 'to'):
                query_embedding = query_embedding.to(device)
            
            # Barcha ma'lumotlarni embedding qilish
            # Agar entity_embeddings_cache mavjud bo'lmasa, yangi embeddinglarni hisoblash
            if not hasattr(self, 'entity_embeddings_cache') or self.entity_embeddings_cache is None:
                logging.info("CKP embeddinglar keshi yaratilmoqda...")
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
            
            # Agar mavjud cache boshqa device da bo'lsa, uni to'g'ri device ga ko'chirish
            if hasattr(self.entity_embeddings_cache, 'device') and self.entity_embeddings_cache.device != device:
                self.entity_embeddings_cache = self.entity_embeddings_cache.to(device)
            
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
                if score > 0.3:  # Minimal o'xshashlik chegarasi
                    # Muhim so'zlar mavjudligini tekshirish - bu natijalarni yaxshilaydi
                    word_match_bonus = 0
                    content = self.entity_infos[idx].get("content", "")
                    for word in important_words:
                        if word in content.lower():
                            word_match_bonus += 0.1
                    
                    # Umumiy o'xshashlik
                    total_sim = score.item() + word_match_bonus
                    results.append((total_sim, self.entity_texts[idx]))
            
            # Natijalarni o'xshashlik darajasi bo'yicha saralash
            results.sort(key=lambda x: x[0], reverse=True)
            
            logging.info(f"Semantik qidiruv natijalar soni: {len(results)}")
            
            if return_raw:
                return [text for _, text in results]
            
            # Natijalarni formatlash
            if results:
                if len(results) > 5:
                    result_texts = []
                    for i, (sim, text) in enumerate(results[:5]):
                        result_texts.append(f"{text} (o'xshashlik: {sim:.2f})")
                    result_text = "\n\n".join(result_texts)
                    result_text += f"\n\nJami {len(results)} ta natija topildi. Qidirishni aniqlashtiring."
                else:
                    result_texts = []
                    for i, (sim, text) in enumerate(results):
                        result_texts.append(f"{text} (o'xshashlik: {sim:.2f})")
                    result_text = "\n\n".join(result_texts)
                
                return result_text
            
            return ""
            
        except Exception as e:
            logging.error(f"Semantik qidirishda xatolik: {str(e)}")
            return [] if return_raw else f"Semantik qidirishda xatolik: {str(e)}"
    
    def _format_results(self, results: List[str]) -> str:
        """Qidiruv natijalarini formatlash
        
        Args:
            results: Natijalar ro'yxati
            
        Returns:
            Formatlangan natija matni
        """
        if not results:
            return ""
        
        formatted_text = ""
        
        # Natijalarni birlashtirish
        formatted_text += "\n\n".join(results)
        
        # Agar natijalar ko'p bo'lsa, qo'shimcha ma'lumot
        if len(results) > 3:
            formatted_text += f"\n\nJami {len(results)} ta natija topildi."
        
        return f"MST ma'lumotlari:\n\n{formatted_text}"
