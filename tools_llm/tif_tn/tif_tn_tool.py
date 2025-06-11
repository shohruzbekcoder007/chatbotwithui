import json
import os
import logging
from typing import List, Dict, Optional, Any, Type, Union
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import util
import torch

from retriever.langchain_chroma import CustomEmbeddingFunction

class TIFTNToolInput(BaseModel):
    query: str = Field(description="Tashqi iqtisodiy faoliyat tovarlar nomenklaturasi (TIF-TN) ma'lumotlarini qidirish uchun so'rov")

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TIFTNTool(BaseTool):
    """TIF TN (Tashqi iqtisodiy faoliyat tovar nomenklaturasi) ma'lumotlari bilan ishlash uchun tool."""
    name: str = "tif_tn_tool"
    description: str = "TIF TN (Tashqi iqtisodiy faoliyat tovar nomenklaturasi) ma'lumotlari bilan ishlash uchun tool. Bu tool orqali TIF-TN ma'lumotlarini qidirish va tahlil qilish mumkin. Masalan: '0102310000' kabi so'rovlar bilan qidiruv qilish mumkin."
    args_schema: Type[BaseModel] = TIFTNToolInput
    
    # Maydonlarni Field orqali e'lon qilish
    json_file_path: str = Field(default="")
    data: List[Dict[str, Any]] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    embedding_model: Any = Field(default=None)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict[str, Any]] = Field(default_factory=list)
    
    def __init__(self, json_file_path: str, use_embeddings: bool = True, embedding_model=None):
        """TifTnTool klassini ishga tushirish.
        
        Args:
            json_file_path: TIF TN ma'lumotlari saqlanadigan JSON fayl yo'li.
            use_embeddings: Embedding modelini ishlatish yoki emasligini belgilash.
            embedding_model: Embedding modelini tashqaridan olish.
        """
        # BaseTool klassining __init__ metodini chaqirish
        super().__init__()
        
        # Maydonlarni o'rnatish
        self.json_file_path = json_file_path
        self.use_embeddings = use_embeddings
        
        # Ma'lumotlarni yuklash
        self.data = self._load_data()
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            logging.info("Tashqaridan berilgan embedding modeli o'rnatildi.")
        else:
            # Embedding modelini kerak bo'lganda yuklaymiz
            self.embedding_model = None
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """JSON fayldan ma'lumotlarni yuklash.
        
        Returns:
            List[Dict[str, Any]]: TIF TN ma'lumotlari ro'yxati.
        """
        try:
            if not os.path.exists(self.json_file_path):
                print(f"Xato: {self.json_file_path} fayli topilmadi.")
                return []
                
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError:
            print(f"Xato: {self.json_file_path} fayli noto'g'ri JSON formatida.")
            return []
        except Exception as e:
            print(f"Xato: {str(e)}")
            return []
    
    def _get_all_items(self) -> List[Dict[str, Any]]:
        """Barcha TIF TN ma'lumotlarini olish.
        
        Returns:
            List[Dict[str, Any]]: Barcha TIF TN ma'lumotlari.
        """
        return self.data
    
    def _search_by_hscode(self, hscode: str) -> Optional[Dict[str, Any]]:
        """HS kod bo'yicha qidirish.
        
        Args:
            hscode: Qidirilayotgan HS kod.
            
        Returns:
            Optional[Dict[str, Any]]: Topilgan ma'lumot yoki None.
        """
        for item in self.data:
            if item.get('hscode') == hscode:
                return item
        return None
    
    def _search_by_chapter(self, chapter_code: str) -> List[Dict[str, Any]]:
        """Bo'lim kodi bo'yicha qidirish.
        
        Args:
            chapter_code: Qidirilayotgan bo'lim kodi.
            
        Returns:
            List[Dict[str, Any]]: Topilgan ma'lumotlar ro'yxati.
        """
        return [item for item in self.data if item.get('chaptercode') == chapter_code]
    
    def _search_by_keyword(self, keyword: str, lang: str = 'uzb') -> List[Dict[str, Any]]:
        """Kalit so'z bo'yicha qidirish.
        
        Args:
            keyword: Qidirilayotgan kalit so'z.
            lang: Til, 'uzb' (o'zbek kirill), 'uzl' (o'zbek lotin), 'rus' (rus) yoki 'eng' (ingliz).
            
        Returns:
            List[Dict[str, Any]]: Topilgan ma'lumotlar ro'yxati.
        """
        keyword = keyword.lower()
        results = []
        
        field_map = {
            'uzb': 'headingnameuzbkiril',
            'uzl': 'headingnameuzblatin',
            'rus': 'headingnamerus',
            'eng': 'headingnameeng'
        }
        
        field = field_map.get(lang, 'headingnameuzbkiril')
        
        for item in self.data:
            if field in item and keyword in item[field].lower():
                results.append(item)
            elif 'shortname' in item and keyword in item['shortname'].lower():
                results.append(item)
                
        return results
    
    def _get_unique_chapters(self) -> List[str]:
        """Noyob bo'lim kodlarini olish.
        
        Returns:
            List[str]: Noyob bo'lim kodlari ro'yxati.
        """
        chapters = set(item.get('chaptercode') for item in self.data if 'chaptercode' in item)
        return sorted(list(chapters))
    
    def _get_units(self) -> Dict[str, List[str]]:
        """Barcha o'lchov birliklarini olish.
        
        Returns:
            Dict[str, List[str]]: Har bir tildagi o'lchov birliklari.
        """
        units = {
            'uzb': set(),
            'rus': set(),
            'eng': set()
        }
        
        for item in self.data:
            if 'unituzbkiril' in item:
                units['uzb'].add(item['unituzbkiril'])
            if 'unitrus' in item:
                units['rus'].add(item['unitrus'])
            if 'uniteng' in item:
                units['eng'].add(item['uniteng'])
                
        return {k: sorted(list(v)) for k, v in units.items()}
    
    def _save_to_json(self, data: List[Dict[str, Any]], output_path: str) -> bool:
        """Ma'lumotlarni JSON faylga saqlash.
        
        Args:
            data: Saqlanadigan ma'lumotlar.
            output_path: Chiqish fayl yo'li.
            
        Returns:
            bool: Saqlash muvaffaqiyatli bo'lsa True, aks holda False.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Saqlashda xatolik: {str(e)}")
            return False
    
    def _prepare_embedding_data(self):
        """Prepare embedding data for the TIF TN data."""
        # Har bir element uchun qidiruv matnini tayyorlash
        self.entity_texts = []
        for entity in self.data:
            # Barcha mavjud tillar uchun matnlarni birlashtirish
            search_text = ""
            if 'headingnameuzbkiril' in entity:
                search_text += entity['headingnameuzbkiril'] + " "
            if 'headingnameuzblatin' in entity:
                search_text += entity['headingnameuzblatin'] + " "
            if 'headingnamerus' in entity:
                search_text += entity['headingnamerus'] + " "
            if 'headingnameeng' in entity:
                search_text += entity['headingnameeng'] + " "
            if 'shortname' in entity:
                search_text += entity['shortname']
                
            self.entity_texts.append(search_text.strip())
        
        logging.info(f"Embedding uchun {len(self.entity_texts)} ta element tayyorlandi.")
        self.entity_infos = self.data

    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # So'rovni qayta ishlash va TIF TN ma'lumotlaridan tegishli ma'lumotni topish
            result_hscode = self._search_by_hscode(query)
            result_chapter = self._search_by_chapter(query)
            result_keyword = self._search_by_keyword(query)
            
            # Natijalarni birlashtirish - dublikatlarni olib tashlash
            all_results = []
            
            # HS kod bo'yicha natija (bitta element bo'lishi mumkin)
            if result_hscode and result_hscode not in all_results:
                all_results.append(result_hscode)
            
            # Bo'lim bo'yicha natijalar (ro'yxat)
            for result in result_chapter:
                if result not in all_results:
                    all_results.append(result)
            
            # Kalit so'z bo'yicha natijalar (ro'yxat)
            for result in result_keyword:
                if result not in all_results:
                    all_results.append(result)
                    
            # Agar oddiy qidiruv natija bermasa va embedding yoqilgan bo'lsa, semantik qidiruvni ishlatish
            if not all_results and self.use_embeddings:
                logging.info(f"Oddiy qidiruv natija bermadi, semantik qidiruv ishlatilmoqda: {query}")
                result_semantic = self._semantic_search(query, return_raw=True)
                
                # Semantik qidiruv natijalarini qo'shish
                for result in result_semantic:
                    if result not in all_results:
                        all_results.append(result)
            else:
                logging.info(f"Oddiy qidiruv natija berdi yoki semantik qidiruv o'chirilgan: {query}")
                # Semantik qidiruv ishlatilmaydi

            # Natijalarni muhim so'zlar asosida saralash
            important_words = [word for word in query.split() if len(word) > 3]
            
            def score_result(result):
                score = 0
                for word in important_words:
                    if 'headingnameuzbkiril' in result and word in result['headingnameuzbkiril'].lower():
                        score += 1
                return score
            
            all_results.sort(key=score_result, reverse=True)

            # Natijalarni formatlash va qaytarish
            if all_results:
                logging.info(f"Hibrid qidiruv muvaffaqiyatli natija berdi: {query}")
                return self._format_results(all_results[:5])
            
            # Natija topilmagan bo'lsa
            return "TIF TN ma'lumotlari topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            
        except Exception as e:
            logging.error(f"TIF TN qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"

    def _semantic_search(self, query: str, return_raw: bool = False) -> Union[List[Dict], str]:
        """Semantik qidiruv."""
        try: 
            # Embedding modelini ishlatish uchun tekshirish
            if not self.use_embeddings:
                return [] if return_raw else "Semantik qidiruv o'chirilgan."
                
            # Embedding ma'lumotlarini tayyorlash
            if not hasattr(self, 'entity_texts') or not hasattr(self, 'entity_infos') or not self.entity_texts:
                self._prepare_embedding_data()
                
            if not self.entity_texts or len(self.entity_texts) == 0:
                return [] if return_raw else "Ma'lumotlar mavjud emas."

            # Embedding modelini tekshirish
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
            
            # Device ni aniqlash (CPU yoki CUDA)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Xotira tejash uchun batch usulida ishlash
            batch_size = 100  # Bir vaqtda qayta ishlanadigan hujjatlar soni
            top_k = 20  # Eng yuqori o'xshashlikdagi natijalar soni
            
            # So'rov embeddingini olish va device ga joylashtirish
            query_embedding = self.embedding_model.encode([query])[0]
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.to(device)
            else:
                query_embedding = torch.tensor(query_embedding).to(device)
            
            # Natijalarni saqlash uchun
            all_scores = []
            
            # Hujjatlarni batch usulida qayta ishlash
            for i in range(0, len(self.entity_texts), batch_size):
                batch_texts = self.entity_texts[i:i+batch_size]
                
                # Batch embeddinglarini olish
                batch_embeddings = self.embedding_model.encode(batch_texts)
                
                # Har bir embedding uchun o'xshashlikni hisoblash - device consistency bilan
                for j, emb in enumerate(batch_embeddings):
                    # Embeddingni to'g'ri device ga ko'chirish
                    if isinstance(emb, torch.Tensor):
                        emb = emb.to(device)
                    else:
                        emb = torch.tensor(emb).to(device)
                    
                    # Cosine similarity hisoblash - tensor dimensionlarini to'g'rilash
                    cos_score = torch.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
                    all_scores.append((i+j, cos_score))
            
            # O'xshashlik bo'yicha saralash
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Top natijalarni olish
            results = []
            for idx, _ in all_scores[:top_k]:
                if idx < len(self.entity_infos):
                    results.append(self.entity_infos[idx])
            
            return results if return_raw else self._format_results(results[:5])
        except Exception as e:
            logging.error(f"Semantik qidirishda xatolik: {str(e)}")
            return [] if return_raw else "Qidirishda xatolik yuz berdi."


    def _format_results(self, results: List[Dict]) -> str:
        """Format qilingan natijalarni qaytarish"""
        if not results:
            return "Ma'lumotlar topilmadi."
            
        formatted_results = ""
        for result in results:
            formatted_results += f"HS kod: {result.get('hscode', 'Mavjud emas')}\n"
            
            if 'headingnameuzbkiril' in result:
                formatted_results += f"Heading name (Uzbkiril): {result['headingnameuzbkiril']}\n"
                
            if 'headingnameuzblatin' in result:
                formatted_results += f"Heading name (Uzblatin): {result['headingnameuzblatin']}\n"
                
            if 'headingnamerus' in result:
                formatted_results += f"Heading name (Rus): {result['headingnamerus']}\n"
                
            if 'headingnameeng' in result:
                formatted_results += f"Heading name (Eng): {result['headingnameeng']}\n"
                
            if 'shortname' in result:
                formatted_results += f"Qisqa nomi: {result['shortname']}\n"
                
            if 'chaptercode' in result:
                formatted_results += f"Bo'lim kodi: {result['chaptercode']}\n"
                
            if 'unituzbkiril' in result:
                formatted_results += f"O'lchov birligi (UZ): {result['unituzbkiril']}\n"
                
            if 'unitrus' in result:
                formatted_results += f"O'lchov birligi (RU): {result['unitrus']}\n"
                
            if 'uniteng' in result:
                formatted_results += f"O'lchov birligi (EN): {result['uniteng']}\n"
                
            formatted_results += "\n"
            
        return formatted_results