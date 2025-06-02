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
            
            # Natijani aniq formatda qaytarish
            return f"SOATO ma'lumotlari:\n\n{result}"
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
            
            # SOATO kodi bo'yicha qidirish
            if re.match(r'^\d+$', query):
                return self._search_by_code(query)
            
            # Gibrid qidiruv - ham oddiy, ham semantik qidiruv
            # 1. Avval oddiy qidiruv bilan tekshiramiz
            text_results = self._search_by_name(query)
            
            # 2. Agar oddiy qidiruv natija bermasa va embedding model mavjud bo'lsa, semantik qidiruvni ishlatamiz
            if text_results and "topilmadi" not in text_results:
                logging.info(f"Oddiy qidiruv natija berdi: {query}")
                return text_results
            elif self.use_embeddings and self.embedding_model is not None:
                logging.info(f"Semantik qidiruv ishlatilmoqda: {query}")
                semantic_results = self._semantic_search(query)
                if semantic_results:
                    return semantic_results
            
            # 3. Agar semantik qidiruv ham natija bermasa, oddiy qidiruv natijasini qaytaramiz
            return text_results
        except Exception as e:
            logging.error(f"SOATO qidirishda xatolik: {str(e)}")
            return f"Qidirishda xatolik yuz berdi: {str(e)}"
    
    def _get_country_info(self) -> str:
        """Mamlakat haqida ma'lumot olish"""
        country = self.soato_data.get("country", {})
        if not country:
            return "Mamlakat haqida ma'lumot topilmadi."
        
        info = f"SOATO kodi: {country.get('code', 'Mavjud emas')}\n"
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
        # Mamlakat bo'yicha qidirish
        country = self.soato_data.get("country", {})
        if country.get("code") == code:
            return self._get_country_info()
        
        # Viloyatlar bo'yicha qidirish
        for region in country.get("regions", []):
            if region.get("code") == code:
                return self._format_region_info(region)
            
            # Tumanlar bo'yicha qidirish
            for district in region.get("districts", []):
                if district.get("code") == code:
                    return self._format_district_info(district, region)
                
                # Shaharlar bo'yicha qidirish
                for city in district.get("cities", []):
                    if city.get("code") == code:
                        return self._format_city_info(city, district, region)
                
                # Shaharchalar bo'yicha qidirish
                for settlement in district.get("urban_settlements", []):
                    if settlement.get("code") == code:
                        return self._format_settlement_info(settlement, district, region)
                
                # Qishloq fuqarolar yig'inlari bo'yicha qidirish
                for assembly in district.get("rural_assemblies", []):
                    if assembly.get("code") == code:
                        return self._format_assembly_info(assembly, district, region)
        
        return f"SOATO kodi {code} bo'yicha ma'lumot topilmadi. Iltimos, boshqa kod bilan qayta urinib ko'ring."
    
    def _semantic_search(self, query: str) -> str:
        """Semantik qidiruv - embedding modelini ishlatib o'xshash ma'lumotlarni topish"""
        if not self.entity_texts or not self.entity_infos:
            return "Embedding ma'lumotlari mavjud emas."
        
        try:
            # Query embedding olish
            query_embedding = self.embedding_model.encode([query], normalize=True)[0]
            entity_embeddings = self.embedding_model.encode(self.entity_texts, normalize=True)
            
            similarities = []
            for i, entity_embedding in enumerate(entity_embeddings):
                similarity = util.cos_sim(query_embedding, entity_embedding).item()
                similarities.append((similarity, i))
            
            similarities.sort(reverse=True)
            top_k = min(5, len(similarities))
            top_entities = []
            
            for i in range(top_k):
                similarity, idx = similarities[i]
                if similarity > 0.3:
                    entity_type, entity, parent, grandparent = self.entity_infos[idx]
                    top_entities.append((entity_type, entity, parent, grandparent, similarity))
            
            if not top_entities:
                return ""
            
            result = f"Semantik qidiruv natijalari:\n\n"
            
            for entity_type, entity, parent, grandparent, similarity in top_entities:
                try:
                    if entity_type == "country":
                        result += self._get_country_info()
                    elif entity_type == "region":
                        result += self._format_region_info(entity)
                    elif entity_type == "district":
                        result += self._format_district_info(entity, parent)
                    elif entity_type == "city":
                        result += self._format_city_info(entity, parent, grandparent)
                    elif entity_type == "urban_settlement":
                        result += self._format_settlement_info(entity, parent, grandparent)
                    elif entity_type == "rural_assembly":
                        result += self._format_assembly_info(entity, parent, grandparent)
                    result += f"O'xshashlik: {similarity:.2f}\n\n"
                except Exception as format_error:
                    logging.error(f"Ma'lumotni formatlashda xatolik: {str(format_error)}")
                    result += f"Ma'lumot formati xatoligi: {entity_type} - {entity.get('name_latin', 'N/A')}\n\n"
            
            return result
        except Exception as e:
            logging.error(f"Semantik qidirishda xatolik: {str(e)}")
            return ""
    
    def _search_by_name(self, query: str) -> str:
        """Nomi bo'yicha ma'muriy hududiy birlikni qidirish"""
        results = []
        country = self.soato_data.get("country", {})
        query = query.lower().strip()
        
        # Qidiruv so'zlarini ajratib olish
        query_words = query.split()
        
        logging.info(f"Qidirilayotgan so'rov: '{query}'")
        
        # Check country name
        country_names = [
            country.get("name_latin", "").lower(),
            country.get("name_cyrillic", "").lower(),
            country.get("name_russian", "").lower()
        ]
        
        # To'liq mos kelish tekshirish
        if any(query == name for name in country_names if name):
            return self._get_country_info()
        # Qismiy mos kelish tekshirish
        elif any(all(word in name for word in query_words) for name in country_names if name):
            return self._get_country_info()
        
        # Search in regions
        for region in country.get("regions", []):
            region_names = [
                region.get("name_latin", "").lower(),
                region.get("name_cyrillic", "").lower(),
                region.get("name_russian", "").lower()
            ]
            
            # To'liq mos kelish tekshirish
            if any(query == name for name in region_names if name):
                results.append(("region", region, None, None))
                continue
            
            # Qismiy mos kelish tekshirish
            if any(all(word in name for word in query_words) for name in region_names if name):
                results.append(("region", region, None, None))
            
            # Search in districts
            for district in region.get("districts", []):
                district_names = [
                    district.get("name_latin", "").lower(),
                    district.get("name_cyrillic", "").lower(),
                    district.get("name_russian", "").lower()
                ]
                
                # "tumani" so'zi bilan qidirish uchun maxsus holat
                district_with_suffix = [f"{name} tumani" for name in district_names if name]
                district_names.extend(district_with_suffix)
                
                # Tuman nomini "tumani" so'zisiz olish
                if "tumani" in query:
                    query_without_suffix = query.replace("tumani", "").strip()
                    
                    # Agar "tumani" so'zi olib tashlangandan keyin tuman nomi to'g'ri kelsa
                    if any(query_without_suffix == name for name in district_names if name):
                        results.append(("district", district, region, None))
                        continue
                
                # To'liq mos kelish tekshirish
                if any(query == name for name in district_names if name):
                    results.append(("district", district, region, None))
                    continue
                
                # Qismiy mos kelish tekshirish
                if any(all(word in name for word in query_words) for name in district_names if name):
                    results.append(("district", district, region, None))
                
                # Search in cities
                for city in district.get("cities", []):
                    city_names = [
                        city.get("name_latin", "").lower(),
                        city.get("name_cyrillic", "").lower(),
                        city.get("name_russian", "").lower()
                    ]
                    
                    # To'liq mos kelish tekshirish
                    if any(query == name for name in city_names if name):
                        results.append(("city", city, district, region))
                        continue
                    
                    # Qismiy mos kelish tekshirish
                    if any(all(word in name for word in query_words) for name in city_names if name):
                        results.append(("city", city, district, region))
                
                # Search in urban settlements
                for settlement in district.get("urban_settlements", []):
                    settlement_names = [
                        settlement.get("name_latin", "").lower(),
                        settlement.get("name_cyrillic", "").lower(),
                        settlement.get("name_russian", "").lower()
                    ]
                    
                    # To'liq mos kelish tekshirish
                    if any(query == name for name in settlement_names if name):
                        results.append(("urban_settlement", settlement, district, region))
                        continue
                    
                    # Qismiy mos kelish tekshirish
                    if any(all(word in name for word in query_words) for name in settlement_names if name):
                        results.append(("urban_settlement", settlement, district, region))
                
                # Search in rural assemblies
                for assembly in district.get("rural_assemblies", []):
                    assembly_names = [
                        assembly.get("name_latin", "").lower(),
                        assembly.get("name_cyrillic", "").lower(),
                        assembly.get("name_russian", "").lower()
                    ]
                    
                    # To'liq mos kelish tekshirish
                    if any(query == name for name in assembly_names if name):
                        results.append(("rural_assembly", assembly, district, region))
                        continue
                    
                    # Qismiy mos kelish tekshirish
                    if any(all(word in name for word in query_words) for name in assembly_names if name):
                        results.append(("rural_assembly", assembly, district, region))
        
        # Natijalarni formatlash
        if not results:
            return f"'{query}' so'rovi bo'yicha ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
        
        result_text = ""
        for entity_type, entity, parent, grandparent in results:
            try:
                if entity_type == "region":
                    result_text += self._format_region_info(entity)
                elif entity_type == "district":
                    result_text += self._format_district_info(entity, parent)
                elif entity_type == "city":
                    result_text += self._format_city_info(entity, parent, grandparent)
                elif entity_type == "urban_settlement":
                    result_text += self._format_settlement_info(entity, parent, grandparent)
                elif entity_type == "rural_assembly":
                    result_text += self._format_assembly_info(entity, parent, grandparent)
                result_text += "\n"
            except Exception as e:
                logging.error(f"Ma'lumotni formatlashda xatolik: {str(e)}")
                result_text += f"Ma'lumot formati xatoligi: {entity_type} - {entity.get('name_latin', 'N/A')}\n\n"
        
        return result_text
    
    def _format_region_info(self, region: Dict) -> str:
        """Viloyat haqida ma'lumotni formatlash"""
        info = f"VILOYAT MA'LUMOTLARI:\n"
        info += f"SOATO kodi: {region.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {region.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {region.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {region.get('name_russian', 'Mavjud emas')}\n"
        info += f"Markaz: {region.get('center_latin', 'Mavjud emas')}\n"
        info += f"Tumanlar soni: {len(region.get('districts', []))}\n"
        
        return info
    
    def _format_district_info(self, district: Dict, region: Dict) -> str:
        """Tuman haqida ma'lumotni formatlash"""
        info = f"TUMAN MA'LUMOTLARI:\n"
        info += f"SOATO kodi: {district.get('code', 'Mavjud emas')}\n"
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
        info += f"SOATO kodi: {city.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {city.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {city.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {city.get('name_russian', 'Mavjud emas')}\n"
        info += f"Tuman: {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        
        return info
    
    def _format_settlement_info(self, settlement: Dict, district: Dict, region: Dict) -> str:
        """Shaharcha haqida ma'lumotni formatlash"""
        info = f"SHAHARCHA MA'LUMOTLARI:\n"
        info += f"SOATO kodi: {settlement.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {settlement.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {settlement.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {settlement.get('name_russian', 'Mavjud emas')}\n"
        info += f"Tuman: {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        
        return info
    
    def _format_assembly_info(self, assembly: Dict, district: Dict, region: Dict) -> str:
        """Qishloq fuqarolar yig'ini haqida ma'lumotni formatlash"""
        info = f"QISHLOQ FUQAROLAR YIG'INI MA'LUMOTLARI:\n"
        info += f"SOATO kodi: {assembly.get('code', 'Mavjud emas')}\n"
        info += f"Nomi (lotin): {assembly.get('name_latin', 'Mavjud emas')}\n"
        info += f"Nomi (kirill): {assembly.get('name_cyrillic', 'Mavjud emas')}\n"
        info += f"Nomi (rus): {assembly.get('name_russian', 'Mavjud emas')}\n"
        info += f"Tuman: {district.get('name_latin', 'Mavjud emas')}\n"
        info += f"Viloyat: {region.get('name_latin', 'Mavjud emas')}\n"
        
        return info
