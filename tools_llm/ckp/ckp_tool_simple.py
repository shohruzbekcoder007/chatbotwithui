import json
import os
import re
import logging
from typing import Dict, List, Any, Optional, Union
from sentence_transformers import SentenceTransformer, util
from retriever.langchain_chroma import CustomEmbeddingFunction

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CkpTool:
    """MST (Mahsulotlarning statistik tasniflagichi) ma'lumotlarini qidirish uchun tool"""
    
    def __init__(self):
        self.name = "ckp_tool"
        self.description = "MST (Mahsulotlarning statistik tasniflagichi) ma'lumotlarini qidirish uchun tool"
        self.ckp_data = []
        self.entity_texts = []
        self.entity_infos = []
        self.embedding_model = None
        
        # Ma'lumotlarni yuklash
        self._load_data()
        
        # Embedding modelini yuklash
        self._load_embedding_model()
        
        # Embedding ma'lumotlarini tayyorlash
        self._prepare_embedding_data()
    
    def _load_data(self):
        """CKP ma'lumotlarini yuklash"""
        try:
            # Ma'lumotlar fayli
            data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckp.json")
            
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
    
    def _load_embedding_model(self):
        """Embedding modelini yuklash"""
        try:
            logging.info("Embedding modeli yuklanmoqda...")
            
            # Custom embedding function orqali modelni yuklash
            embedding_function = CustomEmbeddingFunction()
            
            # Embedding modelini tekshirish
            if hasattr(embedding_function, 'model'):
                self.embedding_model = embedding_function
                logging.info("Embedding modeli muvaffaqiyatli yuklandi.")
            else:
                logging.error("Embedding modeli mavjud emas: CustomEmbeddingFunction.model topilmadi")
                self.embedding_model = None
                
        except Exception as e:
            logging.error(f"Embedding modelini yuklashda xatolik: {str(e)}")
            self.embedding_model = None
    
    def _prepare_embedding_data(self):
        """Embedding uchun ma'lumotlarni tayyorlash"""
        self.entity_texts = []
        self.entity_infos = []
        
        # Barcha ma'lumotlarni qo'shish
        for item in self.ckp_data:
            content = item.get("content", "")
            self.entity_texts.append(content)
            self.entity_infos.append(item)
    
    def run(self, query: str) -> str:
        """Tool ishga tushirilganda bajariladigan asosiy metod
        
        Args:
            query: Foydalanuvchi so'rovi
            
        Returns:
            Formatlangan natija matni
        """
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
                    logging.info(f"Kod qismi bilan mos kelish topildi: {item_code}")
        
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
        
        # Embedding modeli mavjudligini tekshirish
        if self.embedding_model is None:
            error_msg = "Embedding modeli yuklanmagan. Semantik qidiruv ishlamaydi."
            logging.error(error_msg)
            return [] if return_raw else error_msg
        
        try:
            # So'rovni tozalash va muhim so'zlarni ajratib olish
            clean_query = query.lower()
            
            # Muhim so'zlarni ajratib olish
            important_words = [word for word in clean_query.split() if len(word) > 2]
            
            # Query embedding olish
            logging.info(f"Semantik qidiruv ishlatilmoqda: {clean_query}")
            query_embedding = self.embedding_model.encode([clean_query], normalize=True)[0]
            entity_embeddings = self.embedding_model.encode(self.entity_texts, normalize=True)
            
            # O'xshashlik hisoblash
            similarities = []
            for i, entity_embedding in enumerate(entity_embeddings):
                content = self.entity_infos[i].get("content", "")
                
                # Semantik o'xshashlik
                sim = util.dot_score(query_embedding, entity_embedding).item()
                
                # Muhim so'zlar mavjudligini tekshirish - bu natijalarni yaxshilaydi
                word_match_bonus = 0
                for word in important_words:
                    if word in content.lower():
                        word_match_bonus += 0.1
                
                # Umumiy o'xshashlik
                total_sim = sim + word_match_bonus
                similarities.append((i, total_sim, sim))
            
            # Natijalarni saralash
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Eng yuqori o'xshashlikdagi natijalarni olish (0.3 dan yuqori)
            results = []
            for i, total_sim, sim in similarities[:20]:
                if total_sim > 0.3:
                    results.append(self.entity_texts[i])
            
            logging.info(f"Semantik qidiruv natijalar soni: {len(results)}")
            
            if return_raw:
                return results
            
            # Natijalarni formatlash
            if results:
                if len(results) > 5:
                    result_text = "\n\n".join(results[:5])
                    result_text += f"\n\nJami {len(results)} ta natija topildi. Qidirishni aniqlashtiring."
                else:
                    result_text = "\n\n".join(results)
                
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
