import json
import os
import re
import torch
import uuid
import logging
from typing import Dict, List, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

class CustomEmbeddingModel:
    """BAAI/bge-m3 embedding modeli bilan ishlaydigan klass"""
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        print(f"Model yuklanmoqda: {model_name}")
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        
        # Local model papkasini yaratish
        local_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'local_models')
        os.makedirs(local_models_dir, exist_ok=True)
        
        # Model nomini local papka uchun moslashtirish
        model_folder_name = model_name.replace('/', '_')
        local_model_path = os.path.join(local_models_dir, model_folder_name)
        
        # Xotira cheklovlarini sozlash (agar GPU bo'lsa)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)  # GPU xotirasining 95% ini ishlatish
        
        # Logging darajasini pasaytirish
        logging.getLogger("accelerate").setLevel(logging.WARNING)
        
        # Agar model local'da mavjud bo'lsa, local'dan yuklash
        if os.path.exists(local_model_path):
            print(f"Model local'dan yuklanmoqda: {local_model_path}")
            self.model = SentenceTransformer(local_model_path, device=str(self.device))
        else:
            print(f"Model internetdan yuklanmoqda va local'ga saqlanadi: {model_name}")
            self.model = SentenceTransformer(model_name, device=str(self.device))
            # Modelni local'ga saqlash
            self.model.save(local_model_path)
            print(f"Model local'ga saqlandi: {local_model_path}")
        
        self.max_length = 512  # SOATO ma'lumotlari uchun kichikroq uzunlik yetarli
        self.batch_size = 32   # Kichikroq batch size
    
    def encode(self, texts: List[str], normalize=True) -> List[List[float]]:
        """Matnlarni embedding vektorlariga o'zgartirish"""
        # Matnlarni tozalash
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Embedding yaratish
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            max_length=self.max_length,
            truncation=True,
            show_progress_bar=False
        )
        
        return embeddings.tolist() if isinstance(embeddings, torch.Tensor) else embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """Matnni tozalash va formatlash"""
        if not isinstance(text, str):
            text = str(text)
        
        # HTML teglarni olib tashlash
        text = re.sub(r'<[^>]+>', '', text)
        
        # Ko'p bo'sh joylarni bitta bo'sh joy bilan almashtirish
        text = ' '.join(text.split())
        
        return text
    
    def similarity_search(self, query: str, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantik o'xshashlik asosida qidirish"""
        # Query va matnlarni embedding vektorlariga o'zgartirish
        query_embedding = self.encode([query], normalize=True)[0]
        text_embeddings = self.encode(texts, normalize=True)
        
        # Cosine o'xshashlikni hisoblash
        similarities = [util.cos_sim(query_embedding, text_embedding)[0][0].item() for text_embedding in text_embeddings]
        
        # Natijalarni o'xshashlik bo'yicha tartiblash
        results = []
        for i, similarity in enumerate(similarities):
            results.append({
                "index": i,
                "text": texts[i],
                "similarity": similarity
            })
        
        # O'xshashlik bo'yicha kamayish tartibida saralash
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        
        # Top-K natijalarni qaytarish
        return results[:top_k]


class SoatoTool(BaseTool):
    """Tool for querying information about administrative divisions of Uzbekistan using SOATO data."""
    name: str = "soato_tool"
    description: str = """Useful for answering questions about administrative divisions of Uzbekistan.
    This tool can provide information about regions, districts, cities, urban settlements, and rural assemblies.
    You can query by name or code, and get details about administrative centers and hierarchical relationships."""
    soato_data: Dict[str, Any] = None
    embedding_model: Any = None
    use_embeddings: bool = True
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[tuple] = Field(default_factory=list)
    
    def __init__(self, soato_file_path: str, use_embeddings: bool = True):
        """Initialize the SOATO tool with the path to the SOATO JSON file."""
        super().__init__()
        self.soato_data = self._load_soato_data(soato_file_path)
        self.use_embeddings = use_embeddings
        
        # Embedding modelini yuklash (agar use_embeddings=True bo'lsa)
        if self.use_embeddings:
            print("BAAI/bge-m3 embedding modeli yuklanmoqda...")
            self.embedding_model = CustomEmbeddingModel(model_name='BAAI/bge-m3')
            print("Embedding modeli yuklandi.")
            
            # Embedding uchun ma'lumotlarni tayyorlash
            self._prepare_embedding_data()
        
    def _load_soato_data(self, file_path: str) -> Dict[str, Any]:
        """Load SOATO data from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"SOATO data file not found at {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _run(self, query: str) -> str:
        """So'rovni bajarish va natijani qaytarish"""
        # So'rovni qayta ishlash va SOATO ma'lumotlaridan tegishli ma'lumotni topish
        result = self._search_soato_data(query)
        if not result:
            return "Ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
        return result
    
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
        """Search for information in SOATO data based on the query."""
        query = query.lower()
        
        # Check if query is asking about country information
        if any(keyword in query for keyword in ["o'zbekiston", "uzbekistan", "страна", "республика", "country"]):
            return self._get_country_info()
        
        # Check if query is asking for a list of regions
        if any(keyword in query for keyword in ["viloyatlar", "regions", "области", "ro'yxati", "список", "list"]):
            return self._get_regions_list()
        
        # Check if query contains a SOATO code
        if any(char.isdigit() for char in query):
            code = ''.join(char for char in query if char.isdigit())
            return self._search_by_code(code)
        
        # Agar embedding model mavjud bo'lsa, semantik qidiruv
        if self.use_embeddings and self.embedding_model:
            embedding_results = self._semantic_search(query)
            if embedding_results:
                return embedding_results
        
        # Agar semantik qidiruv natija bermasa, oddiy qidiruv
        return self._search_by_name(query)
    
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
        """Hybrid qidiruv - semantik va kalit so'zlar bo'yicha qidiruv kombinatsiyasi"""
        try:
            # 1. Semantik qidiruv - embedding model orqali
            semantic_results = self.embedding_model.similarity_search(query, self.entity_texts, top_k=10)
            
            # 2. Kalit so'zlar bo'yicha qidiruv
            keyword_results = []
            query_words = query.lower().split()
            
            for i, text in enumerate(self.entity_texts):
                # Har bir so'z uchun tekshirish
                matches = sum(1 for word in query_words if word in text.lower())
                if matches > 0:  # Agar kamida bitta so'z topilsa
                    keyword_results.append({
                        "index": i,
                        "text": text,
                        "matches": matches,
                        "similarity": matches / len(query_words)  # Oddiy o'xshashlik o'lchovi
                    })
            
            # Kalit so'z natijalarini saralash
            keyword_results = sorted(keyword_results, key=lambda x: x["similarity"], reverse=True)[:10]
            
            # 3. Natijalarni birlashtirish
            all_results = []
            seen_indices = set()
            
            # Semantik natijalarni qo'shish
            for result in semantic_results:
                if result["index"] not in seen_indices and result["similarity"] >= 0.2:  # Minimal o'xshashlik chegarasi
                    all_results.append((result["index"], result["similarity"] * 0.7))  # Semantik natijalar uchun 0.7 vazn
                    seen_indices.add(result["index"])
            
            # Kalit so'z natijalarni qo'shish
            for result in keyword_results:
                if result["index"] not in seen_indices and result["similarity"] >= 0.2:  # Minimal o'xshashlik chegarasi
                    all_results.append((result["index"], result["similarity"] * 0.3))  # Kalit so'z natijalar uchun 0.3 vazn
                    seen_indices.add(result["index"])
            
            # Natijalarni o'xshashlik bo'yicha saralash
            all_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:5]  # Top-5 natijalar
            
            if not all_results:
                return f"\"{query}\" so'rovi bo'yicha ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
            # Natijalarni formatlash
            results = []
            for idx, similarity in all_results:
                entity_type, entity, parent, grandparent = self.entity_infos[idx]
                
                # Natija formatini tanlash
                if entity_type == "country":
                    results.append((similarity, self._get_country_info()))
                elif entity_type == "region":
                    results.append((similarity, self._format_region_info(entity)))
                elif entity_type == "district":
                    results.append((similarity, self._format_district_info(entity, parent)))
                elif entity_type == "city":
                    results.append((similarity, self._format_city_info(entity, parent, grandparent)))
                elif entity_type == "urban_settlement":
                    results.append((similarity, self._format_settlement_info(entity, parent, grandparent)))
                elif entity_type == "rural_assembly":
                    results.append((similarity, self._format_assembly_info(entity, parent, grandparent)))
            
            # Eng yuqori o'xshashlikdagi natijani qaytarish
            if results:
                return results[0][1]  # Eng yuqori o'xshashlikdagi natijani qaytarish
            
            return f"\"{query}\" so'rovi bo'yicha ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
            
        except Exception as e:
            print(f"Hybrid qidirishda xatolik: {str(e)}")
            return f"Qidirishda xatolik yuz berdi: {str(e)}. Iltimos, boshqacha so'rovni kiriting."
    
    def _search_by_name(self, query: str) -> str:
        """Nomi bo'yicha ma'muriy hududiy birlikni qidirish"""
        results = []
        country = self.soato_data.get("country", {})
        
        # Check country name
        country_names = [
            country.get("name_latin", "").lower(),
            country.get("name_cyrillic", "").lower(),
            country.get("name_russian", "").lower()
        ]
        if any(query in name for name in country_names if name):
            return self._get_country_info()
        
        # Search in regions
        for region in country.get("regions", []):
            region_names = [
                region.get("name_latin", "").lower(),
                region.get("name_cyrillic", "").lower(),
                region.get("name_russian", "").lower()
            ]
            if any(query in name for name in region_names if name):
                results.append(("region", region, None, None))
            
            # Search in districts
            for district in region.get("districts", []):
                district_names = [
                    district.get("name_latin", "").lower(),
                    district.get("name_cyrillic", "").lower(),
                    district.get("name_russian", "").lower()
                ]
                if any(query in name for name in district_names if name):
                    results.append(("district", district, region, None))
                
                # Search in cities
                for city in district.get("cities", []):
                    city_names = [
                        city.get("name_latin", "").lower(),
                        city.get("name_cyrillic", "").lower(),
                        city.get("name_russian", "").lower()
                    ]
                    if any(query in name for name in city_names if name):
                        results.append(("city", city, district, region))
                
                # Search in urban settlements
                for settlement in district.get("urban_settlements", []):
                    settlement_names = [
                        settlement.get("name_latin", "").lower(),
                        settlement.get("name_cyrillic", "").lower(),
                        settlement.get("name_russian", "").lower()
                    ]
                    if any(query in name for name in settlement_names if name):
                        results.append(("urban_settlement", settlement, district, region))
                
                # Search in rural assemblies
                for assembly in district.get("rural_assemblies", []):
                    assembly_names = [
                        assembly.get("name_latin", "").lower(),
                        assembly.get("name_cyrillic", "").lower(),
                        assembly.get("name_russian", "").lower()
                    ]
                    if any(query in name for name in assembly_names if name):
                        results.append(("rural_assembly", assembly, district, region))
        
        if not results:
            return f"\"{query}\" so'rovi bo'yicha ma'lumot topilmadi. Iltimos, boshqacha so'rovni kiriting."
        
        if len(results) == 1:
            entity_type, entity, parent, grandparent = results[0]
            if entity_type == "region":
                return self._format_region_info(entity)
            elif entity_type == "district":
                return self._format_district_info(entity, parent)
            elif entity_type == "city":
                return self._format_city_info(entity, parent, grandparent)
            elif entity_type == "urban_settlement":
                return self._format_settlement_info(entity, parent, grandparent)
            elif entity_type == "rural_assembly":
                return self._format_assembly_info(entity, parent, grandparent)
        
        # Bir nechta natijalar
        result_text = f"\"{query}\" so'rovi bo'yicha {len(results)} ta natija topildi:\n\n"
        for i, (entity_type, entity, parent, _) in enumerate(results, 1):
            name = entity.get("name_latin", "N/A")
            code = entity.get("code", "N/A")
            type_name = {
                "region": "Viloyat",
                "district": "Tuman",
                "city": "Shahar",
                "urban_settlement": "Shaharchа",
                "rural_assembly": "Qishloq"
            }.get(entity_type, entity_type)
            
            parent_name = "" if parent is None else f" ({parent.get('name_latin', 'N/A')})"
            result_text += f"{i}. {type_name}: {name}{parent_name} - SOATO: {code}\n"
        
        return result_text
    
    def _format_region_info(self, region: Dict[str, Any]) -> str:
        """Format information about a region."""
        info = f"Viloyat: {region.get('name_latin', 'N/A')}\n"
        info += f"SOATO kodi: {region.get('code', 'N/A')}\n"
        info += f"Nomi (lotin): {region.get('name_latin', 'N/A')}\n"
        info += f"Nomi (kirill): {region.get('name_cyrillic', 'N/A')}\n"
        info += f"Nomi (rus): {region.get('name_russian', 'N/A')}\n"
        info += f"Markaz: {region.get('center_latin', 'N/A')}\n"
        info += f"Tumanlar soni: {len(region.get('districts', []))}\n"
        
        return info
    
    def _format_district_info(self, district: Dict[str, Any], region: Dict[str, Any]) -> str:
        """Format information about a district."""
        info = f"Tuman: {district.get('name_latin', 'N/A')}\n"
        info += f"Viloyat: {region.get('name_latin', 'N/A')}\n"
        info += f"SOATO kodi: {district.get('code', 'N/A')}\n"
        info += f"Nomi (lotin): {district.get('name_latin', 'N/A')}\n"
        info += f"Nomi (kirill): {district.get('name_cyrillic', 'N/A')}\n"
        info += f"Nomi (rus): {district.get('name_russian', 'N/A')}\n"
        info += f"Markaz: {district.get('center_latin', 'N/A')}\n"
        info += f"Shaharlar soni: {len(district.get('cities', []))}\n"
        info += f"Shaharchalar soni: {len(district.get('urban_settlements', []))}\n"
        info += f"Qishloq fuqarolar yig'inlari soni: {len(district.get('rural_assemblies', []))}\n"
        
        return info
    
    def _format_city_info(self, city: Dict[str, Any], district: Dict[str, Any], region: Dict[str, Any]) -> str:
        """Format information about a city."""
        info = f"Shahar: {city.get('name_latin', 'N/A')}\n"
        info += f"Tuman: {district.get('name_latin', 'N/A')}\n"
        info += f"Viloyat: {region.get('name_latin', 'N/A')}\n"
        info += f"SOATO kodi: {city.get('code', 'N/A')}\n"
        info += f"Nomi (lotin): {city.get('name_latin', 'N/A')}\n"
        info += f"Nomi (kirill): {city.get('name_cyrillic', 'N/A')}\n"
        info += f"Nomi (rus): {city.get('name_russian', 'N/A')}\n"
        
        return info
    
    def _format_settlement_info(self, settlement: Dict[str, Any], district: Dict[str, Any], region: Dict[str, Any]) -> str:
        """Format information about an urban settlement."""
        info = f"Shaharcha: {settlement.get('name_latin', 'N/A')}\n"
        info += f"Tuman: {district.get('name_latin', 'N/A')}\n"
        info += f"Viloyat: {region.get('name_latin', 'N/A')}\n"
        info += f"SOATO kodi: {settlement.get('code', 'N/A')}\n"
        info += f"Nomi (lotin): {settlement.get('name_latin', 'N/A')}\n"
        info += f"Nomi (kirill): {settlement.get('name_cyrillic', 'N/A')}\n"
        info += f"Nomi (rus): {settlement.get('name_russian', 'N/A')}\n"
        
        return info
    
    def _format_assembly_info(self, assembly: Dict[str, Any], district: Dict[str, Any], region: Dict[str, Any]) -> str:
        """Format information about a rural assembly."""
        info = f"Qishloq fuqarolar yig'ini: {assembly.get('name_latin', 'N/A')}\n"
        info += f"Tuman: {district.get('name_latin', 'N/A')}\n"
        info += f"Viloyat: {region.get('name_latin', 'N/A')}\n"
        info += f"SOATO kodi: {assembly.get('code', 'N/A')}\n"
        info += f"Nomi (lotin): {assembly.get('name_latin', 'N/A')}\n"
        info += f"Nomi (kirill): {assembly.get('name_cyrillic', 'N/A')}\n"
        info += f"Nomi (rus): {assembly.get('name_russian', 'N/A')}\n"
        
        return info

# Example usage:
# soato_tool = SoatoTool("path/to/soato.json")
# result = soato_tool.run("Toshkent viloyati")