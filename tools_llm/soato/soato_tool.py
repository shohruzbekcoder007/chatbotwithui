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

# CustomEmbeddingFunction klassi retriever.langchain_chroma modulidan import qilingan

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SoatoTool(BaseTool):
    """SOATO ma'lumotlarini qidirish uchun tool."""
    name: str = "soato_tool"
    description: str = "O'zbekiston Respublikasining ma'muriy-hududiy birliklari (SOATO) ma'lumotlarini qidirish."
    
    soato_data: Dict = Field(default_factory=dict)
    use_embeddings: bool = Field(default=False)
    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    
    def __init__(self, soato_file_path: str, use_embeddings: bool = True):
        """Initialize the SOATO tool with the path to the SOATO JSON file."""
        super().__init__()
        self.soato_data = self._load_soato_data(soato_file_path)
        self.use_embeddings = use_embeddings
        
        # Embedding modelini yaratish
        if self.use_embeddings and self.embedding_model is None:
            print("Embedding modeli yuklanmoqda...")
            self.embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')
            print("Embedding ma'lumotlari tayyorlanmoqda...")
            self._prepare_embedding_data()
        
    def _load_soato_data(self, file_path: str) -> Dict[str, Any]:
        """Load SOATO data from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SOATO data file not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        try:
            # So'rovni qayta ishlash va SOATO ma'lumotlaridan tegishli ma'lumotni topish
            result = self._search_soato_data(query)
            if not result:
                return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Agar natijada "SOATO ma'lumotlari" yoki "MHOBIT ma'lumotlari" qismi bo'lmasa, qo'shish
            if not result.startswith("SOATO ma'lumotlari") and not result.startswith("MHOBIT ma'lumotlari"):
                result = f"SOATO ma'lumotlari:\n\n{result}"
            
            # Natijani aniq formatda qaytarish
            return result
        except Exception as e:
            logging.error(f"SOATO qidirishda xatolik: {str(e)}")
            return f"Ma'lumotlarni qayta ishlashda xatolik yuz berdi: {str(e)}"
    
    def _prepare_embedding_data(self):
        """Embedding uchun ma'lumotlarni tayyorlash"""
        self.entity_texts = []
        self.entity_infos = []
        
        # Mamlakat ma'lumotlarini qo'shish
        country = self.soato_data.get("country", {})
        country_text = f"O'zbekiston Respublikasi {country.get('name_latin', '')} {country.get('name_cyrillic', '')} {country.get('name_russian', '')}"
        self.entity_texts.append(country_text)
        self.entity_infos.append(("country", country, None, None))
        
        # Viloyatlar ma'lumotlarini qo'shish
        for region in country.get("regions", []):
            region_text = f"Viloyat {region.get('name_latin', '')} {region.get('name_cyrillic', '')} {region.get('name_russian', '')} {region.get('center_latin', '')}"
            self.entity_texts.append(region_text)
            self.entity_infos.append(("region", region, None, None))
            
            # Tumanlar ma'lumotlarini qo'shish
            for district in region.get("districts", []):
                district_text = f"Tuman {district.get('name_latin', '')} {district.get('name_cyrillic', '')} {district.get('name_russian', '')} {district.get('center_latin', '')}"
                self.entity_texts.append(district_text)
                self.entity_infos.append(("district", district, region, None))
                
                # Shaharlar ma'lumotlarini qo'shish
                for city in district.get("cities", []):
                    city_text = f"Shahar {city.get('name_latin', '')} {city.get('name_cyrillic', '')} {city.get('name_russian', '')}"
                    self.entity_texts.append(city_text)
                    self.entity_infos.append(("city", city, district, region))
                
                # Shaharchalar ma'lumotlarini qo'shish
                for settlement in district.get("urban_settlements", []):
                    settlement_text = f"Shaharcha {settlement.get('name_latin', '')} {settlement.get('name_cyrillic', '')} {settlement.get('name_russian', '')}"
                    self.entity_texts.append(settlement_text)
                    self.entity_infos.append(("urban_settlement", settlement, district, region))
                
                # Qishloq fuqarolar yig'inlari ma'lumotlarini qo'shish
                for assembly in district.get("rural_assemblies", []):
                    assembly_text = f"Qishloq {assembly.get('name_latin', '')} {assembly.get('name_cyrillic', '')} {assembly.get('name_russian', '')}"
                    self.entity_texts.append(assembly_text)
                    self.entity_infos.append(("rural_assembly", assembly, district, region))
        
        print(f"Embedding uchun {len(self.entity_texts)} ta ma'lumot tayyorlandi.")
    
    def _search_soato_data(self, query: str) -> str:
        """SOATO ma'lumotlaridan so'rov bo'yicha ma'lumot qidirish."""
        try:
            # Agar so'rov bo'sh bo'lsa, umumiy ma'lumot qaytarish
            if not query or query.strip() == "":
                return self._get_country_info()
            
            # So'rovni tayyorlash
            query = query.strip()
            original_query = query
            
            # SOATO va MHOBIT so'zlarini bir xil narsa sifatida qabul qilish
            query = query.lower()
            
            # Bir xil nomli shahar va tumanlarni qidirish
            clean_query = query.lower().replace("shahri", "").replace("tumani", "").replace("shahar", "").replace("tuman", "").strip()
            
            if len(clean_query) >= 3:  # Qidiruv so'zi kamida 3 ta belgidan iborat bo'lishi kerak
                logging.info(f"Bir xil nomli shahar va tumanlarni qidirish: '{clean_query}'")
                
                matching_cities = []
                matching_districts = []
                
                # Barcha viloyatlarni tekshirish
                for region in self.soato_data.get("regions", []):
                    for district in region.get("districts", []):
                        # Tumanni tekshirish
                        district_name = district.get("name_latin", "").lower()
                        if clean_query in district_name:
                            matching_districts.append((district, region))
                        
                        # Shaharlarni tekshirish
                        for city in district.get("cities", []):
                            city_name = city.get("name_latin", "").lower()
                            if clean_query in city_name:
                                matching_cities.append((city, district, region))
                
                # Agar ham shahar, ham tuman topilgan bo'lsa
                if matching_cities and matching_districts:
                    result = "SOATO ma'lumotlari:\n\n"
                    result += "SHAHARLAR:\n"
                    for city, district, region in matching_cities:
                        result += self._format_city_info(city, district, region) + "\n\n"
                    
                    result += "TUMANLAR:\n"
                    for district, region in matching_districts:
                        result += self._format_district_info(district, region) + "\n\n"
                    
                    return result
                # Faqat shaharlar topilgan bo'lsa
                elif matching_cities:
                    if len(matching_cities) == 1:
                        city, district, region = matching_cities[0]
                        return f"SOATO ma'lumotlari:\n{self._format_city_info(city, district, region)}"
                    else:
                        result = "SOATO ma'lumotlari:\n\nSHAHARLAR:\n"
                        for city, district, region in matching_cities:
                            result += self._format_city_info(city, district, region) + "\n\n"
                        return result
                # Faqat tumanlar topilgan bo'lsa
                elif matching_districts:
                    if len(matching_districts) == 1:
                        district, region = matching_districts[0]
                        return f"SOATO ma'lumotlari:\n{self._format_district_info(district, region)}"
                    else:
                        result = "SOATO ma'lumotlari:\n\nTUMANLAR:\n"
                        for district, region in matching_districts:
                            result += self._format_district_info(district, region) + "\n\n"
                        return result
            
            # MHOBIT va SOATO so'zlarini bir xil narsa sifatida qabul qilish va qidiruv so'rovidan olib tashlash
            query = query.replace("mhobit", "")
            query = query.replace("мхобт", "")
            query = query.replace("soato", "")
            query = query.replace("kodi", "").strip()
            
            # Qarshi shahri va boshqa shaharlar/tumanlar yuqoridagi mexanizm orqali qidiriladi
            
            logging.info(f"Qidirilayotgan so'rov: '{original_query}', tozalangan so'rov: '{query}'")
            
            # Original so'rovda raqamli kod qidirish
            code_match = re.search(r'\b(\d{5,10})\b', original_query)
            if code_match:
                code = code_match.group(1)
                logging.info(f"Original so'rovdan SOATO kodi topildi: {code}")
                code_result = self._search_by_code(code)
                if code_result and "topilmadi" not in code_result:
                    logging.info(f"SOATO kodi bo'yicha natija topildi: {code}")
                    return code_result
        
            # Tozalangan so'rovda raqamli kod qidirish
            if re.match(r'^\d+$', query):
                code_result = self._search_by_code(query)
                if code_result and "topilmadi" not in code_result:
                    logging.info(f"SOATO kodi bo'yicha natija topildi: {query}")
                    return code_result
            
            # Agar so'rov juda qisqa bo'lib qolsa (masalan, faqat "kodi" so'zi bo'lsa), original so'rovni qaytaramiz
            if len(query) < 3 and len(original_query) > 3:
                query = original_query
            
            # Gibrid qidiruv strategiyasi
            text_results = None
            semantic_results = None
            
            # 1. Avval oddiy matn qidiruvini sinab ko'ramiz
            text_results = self._search_by_name(query)
            
            # Agar oddiy qidiruv aniq natija bersa, uni qaytaramiz
            if text_results and "topilmadi" not in text_results:
                logging.info(f"Oddiy qidiruv muvaffaqiyatli natija berdi: {query}")
                return f"SOATO ma'lumotlari:\n{text_results}"
            
            # 2. Agar embedding model mavjud bo'lsa, semantik qidiruvni ham sinab ko'ramiz
            if self.use_embeddings and self.embedding_model is not None:
                logging.info(f"Semantik qidiruv ishlatilmoqda: {query}")
                semantic_results = self._semantic_search(query)
                
                # Agar semantik qidiruv natija bersa, uni qaytaramiz
                if semantic_results and "topilmadi" not in semantic_results:
                    logging.info(f"Semantik qidiruv muvaffaqiyatli natija berdi: {query}")
                    return f"SOATO ma'lumotlari:\n{semantic_results}"
            
            # 3. Agar hech qanday qidiruv natija bermasa, eng yaxshi mavjud natijani qaytaramiz
            if text_results and "topilmadi" not in text_results:
                return f"SOATO ma'lumotlari:\n{text_results}"
            elif semantic_results and "topilmadi" not in semantic_results:
                return f"SOATO ma'lumotlari:\n{semantic_results}"
            elif text_results:
                return f"SOATO ma'lumotlari:\n{text_results}"
            else:
                return f"SOATO ma'lumotlari:\n'{query}' so'rovi bo'yicha ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
                
        except Exception as e:
            logging.error(f"SOATO qidirishda xatolik: {str(e)}")
            return f"SOATO ma'lumotlari:\nQidirishda xatolik yuz berdi: {str(e)}"
    
    def _get_country_info(self) -> str:
        """Mamlakat haqida ma'lumot olish"""
        country = self.soato_data.get("country", {})
        if not country:
            return "Mamlakat haqida ma'lumot topilmadi."
        
        info = f"SOATO/MHOBIT kodi: {country.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {country.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {country.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {country.get('name_russian', 'Mavjud emas')}\n"
        info += f"Viloyatlar soni: {len(country.get('regions', []))}\n"
        
        return info
    
    def _get_regions_list(self) -> str:
        """Barcha viloyatlar ro'yxatini olish"""
        regions = self.soato_data.get("country", {}).get("regions", [])
        if not regions:
            return "Viloyatlar ro'yxati topilmadi."
        
        result = "O'zbekiston Respublikasi viloyatlari ro'yxati:\n\n"
        for i, region in enumerate(regions, 1):
            result += f"{i}. {region.get('name_latin', 'Mavjud emas')} (SOATO: {region.get('code', 'Mavjud emas')})\n"
            result += f"   Markaz: {region.get('center_latin', 'Mavjud emas')}\n"
        
        return result
    
    def _search_by_code(self, code: str) -> str:
        """SOATO kodi bo'yicha ma'muriy hududiy birlikni qidirish"""
        # Raqamli kod ekanligini tekshirish va formatni to'g'rilash
        code = code.strip()
        # Raqamli kod bo'lmasa, xato qaytarish
        if not code.isdigit():
            return f"Noto'g'ri SOATO kodi formati: {code}. SOATO kodi faqat raqamlardan iborat bo'lishi kerak."
        
        logging.info(f"SOATO kodi bo'yicha qidirilmoqda: {code}")
        
        # Mamlakat bo'yicha qidirish
        country = self.soato_data.get("country", {})
        if country.get("code") == code:
            return self._get_country_info()
        
        # Viloyatlar bo'yicha qidirish
        for region in country.get("regions", []):
            if str(region.get("code")) == code:
                return self._format_region_info(region)
            
            # Tumanlar bo'yicha qidirish
            for district in region.get("districts", []):
                if str(district.get("code")) == code:
                    return self._format_district_info(district, region)
                
                # Shaharlar bo'yicha qidirish
                for city in district.get("cities", []):
                    if str(city.get("code")) == code:
                        return self._format_city_info(city, district, region)
                
                # Shaharchalar bo'yicha qidirish
                for settlement in district.get("urban_settlements", []):
                    if str(settlement.get("code")) == code:
                        return self._format_settlement_info(settlement, district, region)
                
                # Qishloq fuqarolar yig'inlari bo'yicha qidirish
                for assembly in district.get("rural_assemblies", []):
                    if str(assembly.get("code")) == code:
                        return self._format_assembly_info(assembly, district, region)
        
        # Agar kod topilmasa, qo'shimcha tekshirish - ba'zan kodlar raqam sifatida saqlangan bo'lishi mumkin
        try:
            int_code = int(code)
            # Viloyatlar bo'yicha qidirish
            for region in country.get("regions", []):
                if region.get("code") == int_code:
                    return self._format_region_info(region)
                
                # Tumanlar bo'yicha qidirish
                for district in region.get("districts", []):
                    if district.get("code") == int_code:
                        return self._format_district_info(district, region)
                    
                    # Shaharlar bo'yicha qidirish
                    for city in district.get("cities", []):
                        if city.get("code") == int_code:
                            return self._format_city_info(city, district, region)
                    
                    # Shaharchalar bo'yicha qidirish
                    for settlement in district.get("urban_settlements", []):
                        if settlement.get("code") == int_code:
                            return self._format_settlement_info(settlement, district, region)
                    
                    # Qishloq fuqarolar yig'inlari bo'yicha qidirish
                    for assembly in district.get("rural_assemblies", []):
                        if assembly.get("code") == int_code:
                            return self._format_assembly_info(assembly, district, region)
        except ValueError:
            pass  # Raqamga o'tkazishda xatolik bo'lsa, o'tkazib yuborish
        
        return f"SOATO kodi {code} bo'yicha ma'lumot topilmadi. Iltimos, boshqa kod bilan qayta urinib ko'ring."
    
    def _semantic_search(self, query: str) -> str:
        """Semantik qidiruv - embedding modelini ishlatib o'xshash ma'lumotlarni topish"""
        if not self.entity_texts or not self.entity_infos:
            return "Embedding ma'lumotlari mavjud emas."
        
        try:
            # Query embedding olish
            logging.info(f"Semantik qidiruv ishlatilmoqda: {query}")
            query_embedding = self.embedding_model.encode([query], normalize=True)[0]
            entity_embeddings = self.embedding_model.encode(self.entity_texts, normalize=True)
            
            similarities = []
            for i, entity_embedding in enumerate(entity_embeddings):
                similarity = util.cos_sim(query_embedding, entity_embedding).item()
                similarities.append((similarity, i))
            
            similarities.sort(reverse=True)
            top_k = min(5, len(similarities))  # Eng yaxshi 5 ta natijani olish
            top_entities = []
            
            for i in range(top_k):
                similarity, idx = similarities[i]
                if similarity > 0.3:  # O'xshashlik 30% dan yuqori bo'lishi kerak
                    entity_type, entity, parent, grandparent = self.entity_infos[idx]
                    top_entities.append((entity_type, entity, parent, grandparent, similarity))
            
            if not top_entities:
                return "Semantik qidiruv bo'yicha natija topilmadi."
            
            # Eng yuqori o'xshashlikka ega natijani olish
            top_entity = max(top_entities, key=lambda x: x[4])
            entity_type, entity, parent, grandparent, similarity = top_entity
            top_similarity = similarity
            
            # Eng mos keladigan natijani formatlash
            try:
                # Asosiy natija formati
                if entity_type == "country":
                    result = self._get_country_info()
                elif entity_type == "region":
                    result = self._format_region_info(entity)
                elif entity_type == "district":
                    result = self._format_district_info(entity, parent)
                elif entity_type == "city":
                    result = self._format_city_info(entity, parent, grandparent)
                elif entity_type == "urban_settlement":
                    result = self._format_settlement_info(entity, parent, grandparent)
                elif entity_type == "rural_assembly":
                    result = self._format_assembly_info(entity, parent, grandparent)
                else:
                    # Agar asosiy natija formati aniqlanmasa, boshqa natijalarni ko'rsatish
                    return "Aniq ma'lumot topilmadi."
                
                # Agar asosiy natija o'xshashligi yuqori bo'lsa (>0.6), faqat asosiy natijani qaytarish
                if top_similarity > 0.6:
                    return result
                
                # Agar asosiy natija o'xshashligi past bo'lsa (<=0.6), boshqa natijalarni ham ko'rsatish
                if len(top_entities) > 1:
                    result += "\n\nBoshqa mos natijalar:\n"
                    for i, (e_type, e, p, gp, sim) in enumerate(top_entities[1:], 1):
                        entity_name = ""
                        if e_type == "country":
                            entity_name = "O'zbekiston Respublikasi"
                        elif e_type == "region":
                            entity_name = f"{e.get('name_latin', 'N/A')} viloyati"
                        elif e_type == "district":
                            entity_name = f"{e.get('name_latin', 'N/A')} tumani"
                        elif e_type == "city":
                            entity_name = f"{e.get('name_latin', 'N/A')} shahri"
                        elif e_type == "urban_settlement":
                            entity_name = f"{e.get('name_latin', 'N/A')} shaharchasi"
                        elif e_type == "rural_assembly":
                            entity_name = f"{e.get('name_latin', 'N/A')} qishlog'i"
                        
                        if entity_name:
                            result += f"\n{i}. {entity_name} (o'xshashlik: {sim:.2f})"
                
                return result
            except Exception as format_error:
                logging.error(f"Ma'lumotni formatlashda xatolik: {str(format_error)}")
                return f"Ma'lumot formati xatoligi: {entity_type} - {entity.get('name_latin', 'N/A')}\n\n"
        except Exception as e:
            logging.error(f"Semantik qidirishda xatolik: {str(e)}")
            return "Semantik qidiruv jarayonida xatolik yuz berdi."
    
    def _search_by_name(self, query: str) -> str:
        """Nomi bo'yicha ma'muriy hududiy birlikni qidirish"""
        query = query.lower().strip()
        exact_matches = []
        partial_matches = []
        
        # "tumani", "shahri", "qishlog'i" kabi qo'shimchalar bilan qidirish
        search_type = None
        search_name = query
        
        # MHOBIT va SOATO so'zlarini bir xil ko'rish va qidiruv so'rovidan olib tashlash
        search_name = search_name.replace("mhobit", "")
        search_name = search_name.replace("мхобт", "")
        search_name = search_name.replace("soato", "")
        search_name = search_name.replace("kodi", "").strip()
        
        if "tumani" in search_name:
            search_type = "district"
            search_name = search_name.replace("tumani", "").strip()
        elif "shahri" in search_name:
            search_type = "city"
            search_name = search_name.replace("shahri", "").strip()
        elif "shaharchasi" in search_name:
            search_type = "urban_settlement"
            search_name = search_name.replace("shaharchasi", "").strip()
        elif "qishlog'i" in search_name or "qishlogi" in search_name:
            search_type = "rural_assembly"
            search_name = search_name.replace("qishlog'i", "").replace("qishlogi", "").strip()
        
        logging.info(f"Qidirilayotgan so'rov: '{query}', qidiruv nomi: '{search_name}'")
        
        # Davlat ma'lumotlarini qidirish
        country = self.soato_data.get("country", {})
        country_names = [
            country.get("name_latin", "").lower(),
            country.get("name_cyrillic", "").lower(),
            country.get("name_russian", "").lower()
        ]
        
        if search_type is None or search_type == "country":
            # To'liq mos kelish tekshirish
            if any(search_name == name for name in country_names if name):
                exact_matches.append(("country", self._get_country_info(), 1.0))
            # Qismiy mos kelish tekshirish
            elif any(search_name in name for name in country_names if name):
                partial_matches.append(("country", self._get_country_info(), 0.7))
        
        # Viloyatlarni qidirish
        for region in self.soato_data.get("regions", []):
            region_names = [
                region.get("name_latin", "").lower(),
                region.get("name_cyrillic", "").lower(),
                region.get("name_russian", "").lower()
            ]
            
            if search_type is None or search_type == "region":
                # To'liq mos kelish tekshirish
                if any(search_name == name for name in region_names if name):
                    exact_matches.append(("region", self._format_region_info(region), 1.0))
                # Qismiy mos kelish tekshirish
                elif any(search_name in name for name in region_names if name):
                    partial_matches.append(("region", self._format_region_info(region), 0.8))
            
            # Tumanlarni qidirish
            for district in region.get("districts", []):
                district_names = [
                    district.get("name_latin", "").lower(),
                    district.get("name_cyrillic", "").lower(),
                    district.get("name_russian", "").lower()
                ]
                
                # "tumani" qo'shimchasi bilan qidiruv uchun
                district_names_without_suffix = [name.replace("tumani", "").strip() for name in district_names if name]
                
                if search_type is None or search_type == "district":
                    # To'liq mos kelish tekshirish
                    if any(search_name == name for name in district_names if name) or \
                       any(search_name == name for name in district_names_without_suffix if name):
                        exact_matches.append(("district", self._format_district_info(district, region), 1.0))
                    # Qismiy mos kelish tekshirish
                    elif any(search_name in name for name in district_names if name):
                        partial_matches.append(("district", self._format_district_info(district, region), 0.7))
                
                # Shaharlarni qidirish
                for city in district.get("cities", []):
                    city_names = [
                        city.get("name_latin", "").lower(),
                        city.get("name_cyrillic", "").lower(),
                        city.get("name_russian", "").lower()
                    ]
                    
                    # "shahri" qo'shimchasi bilan qidiruv uchun
                    city_names_without_suffix = [name.replace("shahri", "").strip() for name in city_names if name]
                    
                    if search_type is None or search_type == "city":
                        # To'liq mos kelish tekshirish
                        if any(search_name == name for name in city_names if name) or \
                           any(search_name == name for name in city_names_without_suffix if name):
                            exact_matches.append(("city", self._format_city_info(city, district, region), 1.0))
                        # Qismiy mos kelish tekshirish
                        elif any(search_name in name for name in city_names if name):
                            partial_matches.append(("city", self._format_city_info(city, district, region), 0.8))
            
            # Shaharchalarni qidirish
            for settlement in district.get("urban_settlements", []):
                settlement_names = [
                    settlement.get("name_latin", "").lower(),
                    settlement.get("name_cyrillic", "").lower(),
                    settlement.get("name_russian", "").lower()
                ]
                
                if search_type is None or search_type == "urban_settlement":
                    # To'liq mos kelish tekshirish
                    if any(search_name == name for name in settlement_names if name):
                        exact_matches.append(("urban_settlement", self._format_settlement_info(settlement, district, region), 1.0))
                    # Qismiy mos kelish tekshirish
                    elif any(search_name in name for name in settlement_names if name):
                        partial_matches.append(("urban_settlement", self._format_settlement_info(settlement, district, region), 0.7))
            
            # Qishloq fuqarolar yig'inlarini qidirish
            for assembly in district.get("rural_assemblies", []):
                assembly_names = [
                    assembly.get("name_latin", "").lower(),
                    assembly.get("name_cyrillic", "").lower(),
                    assembly.get("name_russian", "").lower()
                ]
                
                if search_type is None or search_type == "rural_assembly":
                    # To'liq mos kelish tekshirish
                    if any(search_name == name for name in assembly_names if name):
                        exact_matches.append(("rural_assembly", self._format_assembly_info(assembly, district, region), 1.0))
                    # Qismiy mos kelish tekshirish
                    elif any(search_name in name for name in assembly_names if name):
                        partial_matches.append(("rural_assembly", self._format_assembly_info(assembly, district, region), 0.7))
    
        # Natijalarni birlashtirish - avval aniq mosliklar, keyin qisman mosliklar
        all_results = exact_matches + partial_matches
        
        if not all_results:
            return f"'{query}' so'rovi bo'yicha ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
        
        # Natijalarni o'xshashlik bo'yicha tartiblash
        all_results.sort(key=lambda x: x[2], reverse=True)
        
        # Eng yuqori natijani olish
        top_result = all_results[0][1]
        
        # Qolgan natijalarni qo'shish
        if len(all_results) > 1:
            top_result += "\n\nBoshqa mos natijalar:\n"
            for i, (entity_type, info, similarity) in enumerate(all_results[1:4], 1):  # Faqat keyingi 3 ta natija
                entity_name = ""
                if "SOATO kodi:" in info:
                    # Info dan nom qismini ajratib olish
                    name_line = [line for line in info.split("\n") if "Nomi (lotin):" in line]
                    if name_line:
                        entity_name = name_line[0].replace("Nomi (lotin):", "").strip()
                
                if entity_name:
                    top_result += f"\n{i}. {entity_name} - {similarity:.2f}"
        
        return top_result
    
    def _format_region_info(self, region: Dict) -> str:
        """Viloyat haqida ma'lumotni formatlash"""
        info = f"VILOYAT MA'LUMOTLARI:\n"
        info += f"SOATO/MHOBIT kodi: {region.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {region.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {region.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {region.get('name_russian', 'Mavjud emas')}\n"
        info += f"Markaz: {region.get('center_latin', 'Mavjud emas')}\n"
        info += f"Tumanlar soni: {len(region.get('districts', []))}\n"
        
        return info
    
    def _format_district_info(self, district: Dict, region: Dict) -> str:
        """Tuman haqida ma'lumotni formatlash"""
        info = f"TUMAN MA'LUMOTLARI:\n"
        info += f"SOATO/MHOBIT kodi: {district.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {district.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {district.get('name_russian', 'Mavjud emas')}\n"
        info += f"Markaz: {district.get('center_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        info += f"Shaharlar soni: {len(district.get('cities', []))}\n"
        info += f"Shaharchalar soni: {len(district.get('urban_settlements', []))}\n"
        info += f"Qishloq fuqarolar yig'inlari soni: {len(district.get('rural_assemblies', []))}\n"
        
        return info
    
    def _format_city_info(self, city: Dict, district: Dict, region: Dict) -> str:
        """Shahar haqida ma'lumotni formatlash"""
        info = f"SHAHAR MA'LUMOTLARI:\n"
        info += f"SOATO/MHOBIT kodi: {city.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {city.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {city.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {city.get('name_russian', 'Mavjud emas')}\n"
        info += f"Tuman: {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        
        return info
    
    def _format_settlement_info(self, settlement: Dict, district: Dict, region: Dict) -> str:
        """Shaharcha haqida ma'lumotni formatlash"""
        info = f"SHAHARCHA MA'LUMOTLARI:\n"
        info += f"SOATO/MHOBIT kodi: {settlement.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {settlement.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {settlement.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {settlement.get('name_russian', 'Mavjud emas')}\n"
        info += f"Tuman: {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        
        return info
    
    def _format_assembly_info(self, assembly: Dict, district: Dict, region: Dict) -> str:
        """Qishloq fuqarolar yig'ini haqida ma'lumotni formatlash"""
        info = f"QISHLOQ FUQAROLAR YIG'INI MA'LUMOTLARI:\n"
        info += f"SOATO/MHOBIT kodi: {assembly.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {assembly.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {assembly.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {assembly.get('name_russian', 'Mavjud emas')}\n"
        info += f"Tuman: {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        
        return info
