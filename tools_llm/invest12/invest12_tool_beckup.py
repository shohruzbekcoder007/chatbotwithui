import json
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, List, Union, Any
import torch
import torch.nn.functional as F
import networkx as nx  # Knowledge Graph uchun NetworkX kutubxonasi
from collections import defaultdict  # Murakkab ma'lumotlar tuzilmalari uchun

# Import CustomEmbeddingFunction tipini to'g'ri yo'l bilan
from retriever.langchain_chroma import CustomEmbeddingFunction

# Logging sozlamalari (o'zbek tilida)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Invest12ToolInput(BaseModel):
    so_rov: str = Field(description="12-invest shakli ma'lumotlarini qidirish uchun so'rov.")

class Invest12Tool(BaseTool):
    """12-invest shakli ma'lumotlarini qidirish va tahlil qilish uchun vosita."""
    
    name: str = "invest12_tool"
    description: str = "12-invest shakli hisoboti ma'lumotlarini qidirish va tahlil qilish uchun vosita."
    args_schema: Type[BaseModel] = Invest12ToolInput

    embedding_model: Optional['CustomEmbeddingFunction'] = Field(default=None)
    invest12_data: Dict = Field(default_factory=dict)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)
    
    # Knowledge Graph bilan bog'liq yangi maydonlar
    use_knowledge_graph: bool = Field(default=False)
    knowledge_graph: Optional[nx.DiGraph] = Field(default=None)
    node_mapping: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, transport4_file_path: str, use_embeddings: bool = True, embedding_model=None, use_knowledge_graph: bool = False):
        super().__init__()
        self.invest12_data = self._load_transport4_data(transport4_file_path)
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        self.use_knowledge_graph = use_knowledge_graph
        
        if embedding_model is not None:
            self._prepare_embedding_data()
            
        if use_knowledge_graph:
            self._build_knowledge_graph()

    def _load_transport4_data(self, transport4_file_path: str) -> Dict:
        """12-invest ma'lumotlarini yuklash."""
        try:
            with open(transport4_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info("12-invest ma'lumotlari muvaffaqiyatli yuklandi")
            return data
        except Exception as e:
            logging.error(f"Ma'lumotlarni yuklashda xatolik: {str(e)}")
            raise

    def _prepare_embedding_data(self):
        """Ma'lumotlarni embedding uchun tayyorlash."""
        self.entity_texts = []
        self.entity_infos = []
        
        for item in self.invest12_data:
            if 'report_form' in item:
                report_form = item['report_form']
                text = f"12-invest shakli: {report_form.get('name', '')}. {report_form.get('submitters', '')}. {report_form.get('structure', '')}. {report_form.get('indicators', '')}"
                self.entity_texts.append(text)
                self.entity_infos.append({
                    'turi': 'hisobot_shakli',
                    'mazmun': report_form,
                    'bo\'lim_turi': report_form.get('section_type', '')
                })
                
            if 'sections' in item:
                for section in item['sections']:
                    section_text = f"{section.get('bob', '')}. {section.get('title', '')}. {section.get('description', '')}"
                    self.entity_texts.append(section_text)
                    self.entity_infos.append({
                        'turi': 'bo\'lim',
                        'mazmun': {
                            'bob': section.get('bob', ''),
                            'sarlavha': section.get('title', ''),
                            'tavsifi': section.get('description', '')
                        },
                        'bo\'lim_turi': section.get('section_type', '')
                    })
                    
                    # Bobdagi ustunlar uchun alohida
                    if 'columns' in section:
                        for column in section['columns']:
                            col_num = column.get('column', '')
                            description = column.get('description', '')
                            
                            # Ustun uchun alohida format - qator kodsiz
                            ustun_text = f"{section.get('bob', '')} {col_num}-ustun: {description}"
                            self.entity_texts.append(ustun_text)
                            self.entity_infos.append({
                                'turi': 'ustun',
                                'mazmun': {
                                    'bob': section.get('bob', ''),
                                    'ustun': col_num,
                                    'tavsifi': description,
                                    'sarlavha': section.get('title', '')
                                },
                                'bo\'lim_turi': section.get('section_type', '')
                            })
                            
                            # Faqat ustun nomi uchun alohida format
                            ustun_nom_text = f"{col_num}-ustun {section.get('bob', '')}: {description}"
                            self.entity_texts.append(ustun_nom_text)
                            self.entity_infos.append({
                                'turi': 'ustun_nom',
                                'mazmun': {
                                    'bob': section.get('bob', ''),
                                    'ustun': col_num,
                                    'tavsifi': description,
                                    'sarlavha': section.get('title', '')
                                },
                                'bo\'lim_turi': section.get('section_type', '')
                            })
                    
                    if 'rows' in section:
                        for row in section['rows']:
                            row_code = row.get('row_code', '')
                            if 'columns' in row:
                                for column in row['columns']:
                                    col_num = column.get('column', '')
                                    description = column.get('description', '')
                                    col_text = f"{section.get('bob', '')} {row_code}-qator {col_num}-ustun: {description}"
                                    self.entity_texts.append(col_text)
                                    self.entity_infos.append({
                                        'turi': 'ustun',
                                        'mazmun': {
                                            'bob': section.get('bob', ''),
                                            'qator_kodi': row_code,
                                            'ustun': col_num,
                                            'tavsifi': description,
                                            'sarlavha': section.get('title', '')
                                        },
                                        'bo\'lim_turi': section.get('section_type', '')
                                    })
                                    
                                    strict_controls = column.get('strict_logical_controls', [])
                                    non_strict_controls = column.get('non_strict_logical_controls', [])
                                    
                                    for control in strict_controls:
                                        control_text = f"{section.get('bob', '')} {row_code}-qator {col_num}-ustun qat'iy nazorat: {control}"
                                        self.entity_texts.append(control_text)
                                        self.entity_infos.append({
                                            'turi': 'qatiy_nazorat',
                                            'mazmun': {
                                                'bob': section.get('bob', ''),
                                                'qator_kodi': row_code,
                                                'ustun': col_num,
                                                'sarlavha': section.get('title', ''),
                                                'nazorat': control
                                            },
                                            'bo\'lim_turi': section.get('section_type', '')
                                        })
                                    
                                    for control in non_strict_controls:
                                        control_text = f"{section.get('bob', '')} {row_code}-qator {col_num}-ustun qat'iy bo'lmagan nazorat: {control}"
                                        self.entity_texts.append(control_text)
                                        self.entity_infos.append({
                                            'turi': 'qatiy_bolmagan_nazorat',
                                            'mazmun': {
                                                'bob': section.get('bob', ''),
                                                'qator_kodi': row_code,
                                                'ustun': col_num,
                                                'sarlavha': section.get('title', ''),
                                                'nazorat': control
                                            },
                                            'bo\'lim_turi': section.get('section_type', '')
                                        })
    
        self.entity_embeddings_cache = None
        logging.info(f"Embedding uchun {len(self.entity_texts)} ta ma'lumot tayyorlandi")

    def _extract_important_words(self, so_rov: str) -> List[str]:
        """So'rovdan muhim so'zlarni ajratib olish."""
        stop_words = ['va', 'bilan', 'uchun', 'qanday', 'qaysi', 'nima', 'qanaqa', 'deb', 'nomalandi', 'nomlanadi', 'nomi', 'tavsifi', 'description']
        words = [word for word in so_rov.lower().split() if word not in stop_words and len(word) > 1]
        return words

    def _search_by_text(self, so_rov: str, return_raw: bool = False) -> Union[List[Dict], Dict]:
        """Matn bo'yicha qidirish."""
        important_words = self._extract_important_words(so_rov)
        logging.info(f"Muhim so'zlar: {important_words}")
        results = []
        
        so_rov_parts = so_rov.lower().split()
        qator_part = None
        ustun_part = None
        bob_part = None
        
        # Bob va ustun qismlarini aniqlash (regex yordamida)
        import re
        
        # Bob qismini aniqlash
        bob_pattern = re.compile(r'(\d+)[-\s]*bob')
        bob_match = bob_pattern.search(so_rov.lower())
        if bob_match:
            bob_part = bob_match.group(1)
            logging.info(f"Bob qismi aniqlandi: {bob_part}")
        
        # Qator/satr qismini aniqlash
        qator_pattern = re.compile(r'(\d+)[-\s]*(qator|satr)')
        qator_match = qator_pattern.search(so_rov.lower())
        if qator_match:
            qator_part = qator_match.group(1)
            logging.info(f"Qator qismi aniqlandi: {qator_part}")
        
        # Ustun qismini aniqlash
        ustun_pattern = re.compile(r'([a-zA-Zа-яА-ЯёЁ\d]+)[-\s]*ustun')
        ustun_match = ustun_pattern.search(so_rov.lower())
        if ustun_match:
            ustun_part = ustun_match.group(1).upper()
            logging.info(f"Ustun qismi aniqlandi: {ustun_part}")
        
        # Agar ustun_part aniqlanmagan bo'lsa, so'rovda alohida harflarni qidirish
        if not ustun_part:
            for part in so_rov_parts:
                if len(part) == 1: # and part.isalpha():
                    ustun_part = part.upper()
                    logging.info(f"Alohida harf sifatida ustun qismi aniqlandi: {ustun_part}")
                    break
        
        # Agar ustun_part aniqlanmagan bo'lsa, raqamlarni tekshirish
        if not ustun_part:
            # "3-bob 3" kabi so'rovlarni tekshirish
            if bob_part:
                for i, part in enumerate(so_rov_parts):
                    if part.isdigit() and ("bob" not in so_rov_parts[i-1] if i > 0 else True):
                        ustun_part = part
                        logging.info(f"Raqam sifatida ustun qismi aniqlandi: {ustun_part}")
                        break

        # Qidiruv natijalarini saqlash uchun
        found_bob = False
        found_qator = False
        found_ustun = False
        
        # Bob va ustun aniq ko'rsatilgan bo'lsa, faqat shu bob va ustun uchun natijalarni qaytarish
        exact_bob_ustun_search = bob_part and ustun_part
        exact_match_results = []
        
        for item in self.invest12_data:
            if 'report_form' in item:
                report_form = item['report_form']
                if so_rov.lower() in report_form.get('name', '').lower() or so_rov.lower() in report_form.get('submitters', '').lower():
                    results.append({
                        'nomi': report_form.get('name', ''),
                        'kodi': '',
                        'tavsifi': report_form.get('submitters', ''),
                        'bo\'lim_turi': 'hisobot_shakli',
                        'tarkibi': report_form.get('structure', ''),
                        'ko\'rsatkichlari': report_form.get('indicators', '')
                    })
        
            if 'sections' in item:
                for section in item['sections']:
                    bob = section.get('bob', '')
                    sarlavha = section.get('title', '')
                    tavsifi = section.get('description', '')
                    
                    # Bob qismini tekshirish
                    bob_match = False
                    if bob_part:
                        bob_match = bob_part in bob
                        if bob_match:
                            found_bob = True
                    
                    if so_rov.lower() in bob.lower() or so_rov.lower() in sarlavha.lower() or bob_match:
                        results.append({
                            'nomi': bob,
                            'kodi': '',
                            'tavsifi': tavsifi,
                            'bo\'lim_turi': section.get('section_type', ''),
                            'bob': bob,
                            'sarlavha': sarlavha
                        })
                
                    # Statistic section_type uchun
                    if section.get('section_type', '') == 'statistic' and 'rows' in section:
                        for row in section['rows']:
                            row_code = row.get('row_code', '')
                            
                            # Qator qismini tekshirish
                            qator_match = False
                            if qator_part:
                                qator_match = qator_part == row_code
                                if qator_match:
                                    found_qator = True
                            
                            if 'columns' in row:
                                for column in row['columns']:
                                    column_code = column.get('column', '')
                                    column_description = column.get('description', '')
                                    
                                    # Ustun qismini tekshirish
                                    ustun_match = False
                                    if ustun_part:
                                        ustun_match = ustun_part == column_code
                                        if ustun_match:
                                            found_ustun = True
                                    
                                    # So'rovda qator va ustun bo'lsa
                                    if (qator_match and ustun_match) or \
                                       (qator_match and not ustun_part) or \
                                       (ustun_match and not qator_part) or \
                                       (so_rov.lower() in column_description.lower()):
                                        
                                        result = {
                                            'nomi': f"{bob} {row_code}-qator {column_code}-ustun",
                                            'kodi': column_code,
                                            'tavsifi': column_description,
                                            'bo\'lim_turi': 'statistic',
                                            'bob': bob,
                                            'qator_kodi': row_code,
                                            'ustun': column_code,
                                            'sarlavha': sarlavha
                                        }
                                        
                                        results.append(result)
                                        
                                        # Aniq bob va ustun qidiruvida mos kelsa
                                        if exact_bob_ustun_search and bob_part in bob and ustun_part == column_code:
                                            exact_match_results.append(result)
                    
                    # Dynamic section_type uchun
                    if section.get('section_type', '') == 'dynamic' and 'columns' in section:
                        # Bob qismini tekshirish
                        bob_section_match = False
                        if bob_part:
                            bob_section_match = bob_part in bob
                            if not bob_section_match and exact_bob_ustun_search:
                                continue
                                
                        for column in section['columns']:
                            ustun_num = column.get('column', '')
                            ustun_tavsifi = column.get('description', '')
                            
                            # Ustun qismini tekshirish - harf yoki raqam bo'lishi mumkin
                            ustun_match = False
                            if ustun_part and ustun_num:
                                # Ustun qismi harf yoki raqam bo'lishi mumkin
                                ustun_match = ustun_part.upper() == ustun_num.upper()
                                if ustun_match:
                                    found_ustun = True
                                    logging.info(f"Dynamic bo'limda ustun mos keldi: {ustun_num}")
                            
                            # So'rovda bob va ustun bo'lsa yoki tavsifda so'rov bo'lsa
                            if (so_rov.lower() in ustun_tavsifi.lower()) or \
                               (ustun_match) or \
                               (bob_part and ustun_part and bob_part in bob and ustun_part.upper() == ustun_num.upper()) or \
                               (bob_part and not ustun_part and bob_part in bob):
                                
                                column_info = {
                                    'nomi': f"{sarlavha}",
                                    'kodi': ustun_num,
                                    'tavsifi': ustun_tavsifi,
                                    'bo\'lim_turi': 'dynamic',
                                    'bob': bob,
                                    'ustun': ustun_num,
                                    'sarlavha': sarlavha,
                                    'tur': 'ustun'
                                }
                                
                                results.append(column_info)
                                
                                # Aniq bob va ustun qidiruvida mos kelsa
                                if exact_bob_ustun_search and bob_part in bob and ustun_part.upper() == ustun_num.upper():
                                    exact_match_results.append(column_info)
                                
                                # Ustun nomi uchun alohida qo'shish
                                results.append({
                                    'nomi': f"{sarlavha}",
                                    'kodi': ustun_num,
                                    'tavsifi': f"{ustun_num}-ustun {bob}: {ustun_tavsifi}",
                                    'bo\'lim_turi': 'dynamic',
                                    'bob': bob,
                                    'ustun': ustun_num,
                                    'sarlavha': sarlavha,
                                    'tur': 'ustun_nom'
                                })
                                
                                # Qat'iy nazoratlarni qo'shish
                                strict_controls = column.get('strict_logical_controls', [])
                                if strict_controls:
                                    for control in strict_controls:
                                        control_info = {
                                            'nomi': f"{bob} - Ustun {ustun_num}",
                                            'kodi': ustun_num,
                                            'tavsifi': control,
                                            'bo\'lim_turi': 'dynamic',
                                            'bob': bob,
                                            'ustun': ustun_num,
                                            'sarlavha': sarlavha,
                                            'tur': 'qatiy_nazorat'
                                        }
                                        results.append(control_info)
                                        
                                        # Aniq bob va ustun qidiruvida mos kelsa
                                        if exact_bob_ustun_search and bob_part in bob and ustun_part.upper() == ustun_num.upper():
                                            exact_match_results.append(control_info)
                                
                                # Qat'iy bo'lmagan nazoratlarni qo'shish
                                non_strict_controls = column.get('non_strict_logical_controls', [])
                                if non_strict_controls:
                                    for control in non_strict_controls:
                                        control_info = {
                                            'nomi': f"{bob} - Ustun {ustun_num}",
                                            'kodi': ustun_num,
                                            'tavsifi': control,
                                            'bo\'lim_turi': 'dynamic',
                                            'bob': bob,
                                            'ustun': ustun_num,
                                            'sarlavha': sarlavha,
                                            'tur': 'qatiy_bolmagan_nazorat'
                                        }
                                        results.append(control_info)
                                        
                                        # Aniq bob va ustun qidiruvida mos kelsa
                                        if exact_bob_ustun_search and bob_part in bob and ustun_part.upper() == ustun_num.upper():
                                            exact_match_results.append(control_info)
        
        # Agar aniq bob va ustun qidiruvi bo'lsa va natijalar topilgan bo'lsa
        if exact_bob_ustun_search and exact_match_results:
            logging.info(f"Aniq bob ({bob_part}) va ustun ({ustun_part}) uchun {len(exact_match_results)} ta natija topildi")
            return exact_match_results if return_raw else self._format_results(exact_match_results)
        
        # Agar aniq bob va ustun qidiruvi bo'lsa, lekin natijalar topilmagan bo'lsa
        if exact_bob_ustun_search and not exact_match_results:
            logging.info(f"Aniq bob ({bob_part}) va ustun ({ustun_part}) uchun natija topilmadi")
            if not return_raw:
                return {
                    "holat": "natijalar_yoq", 
                    "xabar": f"{bob_part}-BOB {ustun_part}-ustun uchun ma'lumot topilmadi", 
                    "soni": 0, 
                    "natijalar": []
                }
        
        # Agar natijalar bo'sh bo'lsa, so'rovni qismlarga bo'lib qidirish
        if not results and len(important_words) > 1:
            for word in important_words:
                word_results = self._search_by_text(word, return_raw=True)
                if word_results:
                    results.extend(word_results)
        
        return results if return_raw else self._format_results(results)

    def _semantic_search(self, so_rov: str, return_raw: bool = False) -> Union[List[Dict], Dict]:
        """Semantik qidiruv."""
        if not self.entity_texts or not self.entity_infos:
            logging.info("Ma'lumotlar mavjud emas")
            return [] if return_raw else {"holat": "natijalar_yoq", "soni": 0, "natijalar": []}
        
        if not self.use_embeddings:
            logging.info("Semantik qidiruv o'chirilgan")
            return [] if return_raw else {"holat": "natijalar_yoq", "soni": 0, "natijalar": []}

        if self.embedding_model is None:
            logging.error("Embedding modeli yuklanmagan")
            return [] if return_raw else {"holat": "xatolik", "xabar": "Embedding modeli yuklanmagan", "natijalar": []}

        so_rov_lower = so_rov.lower()
        if "stir" in so_rov_lower:
            so_rov_lower = so_rov_lower.replace("stir", "").strip()
        if "ktut" in so_rov_lower:
            so_rov_lower = so_rov_lower.replace("ktut", "").strip()

        try:
            so_rov_embedding = self.embedding_model.encode([so_rov_lower], normalize=True)[0]
            if self.entity_embeddings_cache is None:
                self.entity_embeddings_cache = self.embedding_model.encode(self.entity_texts)
            
            so_rov_embedding_tensor = torch.tensor(so_rov_embedding)
            entity_embeddings_tensor = torch.tensor(self.entity_embeddings_cache)
            
            similarities = F.cosine_similarity(so_rov_embedding_tensor.unsqueeze(0), entity_embeddings_tensor, dim=1)
            top_k = 10
            top_indices = torch.argsort(similarities, descending=True)[:top_k]
            
            results = []
            for idx in top_indices:
                similarity = float(similarities[idx])
                if similarity > 0.5:
                    entity_info = self.entity_infos[idx]
                    results.append({
                        'nomi': entity_info.get('mazmun', {}).get('sarlavha', '') or entity_info.get('mazmun', {}).get('tavsifi', ''),
                        'kodi': entity_info.get('mazmun', {}).get('qator_kodi', '') or entity_info.get('mazmun', {}).get('ustun', ''),
                        'tavsifi': self.entity_texts[idx],
                        'bo\'lim_turi': entity_info.get('bo\'lim_turi', ''),
                        'bob': entity_info.get('mazmun', {}).get('bob', ''),
                        'tur': entity_info.get('turi', ''),
                        'o\'xshashlik': similarity
                    })
            
            return results if return_raw else self._format_results(results)
        except Exception as e:
            logging.error(f"Semantik qidiruvda xatolik: {str(e)}")
            return [] if return_raw else {"holat": "xatolik", "xabar": str(e), "natijalar": []}

    def _build_knowledge_graph(self):
        """12-invest ma'lumotlaridan Knowledge Graph yaratish."""
        logging.info("Knowledge Graph yaratilmoqda...")
        
        # NetworkX kutubxonasi yordamida yo'naltirilgan graf yaratish
        self.knowledge_graph = nx.DiGraph()
        self.node_mapping = {}
        
        # Hisobot shakli uchun asosiy tugun
        report_node_id = "report_form"
        self.knowledge_graph.add_node(report_node_id, type="report_form", name="12-invest shakli")
        self.node_mapping[report_node_id] = report_node_id
        
        for item in self.invest12_data:
            # Hisobot shakli ma'lumotlari
            if 'report_form' in item:
                report_form = item['report_form']
                for key, value in report_form.items():
                    if key != 'name' and isinstance(value, str):  # Asosiy xususiyatlarni qo'shish
                        attr_node_id = f"report_attr_{key}"
                        self.knowledge_graph.add_node(attr_node_id, type="report_attribute", name=key, value=value)
                        self.knowledge_graph.add_edge(report_node_id, attr_node_id, relation="has_attribute")
                        self.node_mapping[attr_node_id] = attr_node_id
            
            # Bo'limlar
            if 'sections' in item:
                for section in item['sections']:
                    section_id = f"section_{section.get('bob', '')}"
                    section_title = section.get('title', '')
                    
                    # Bo'lim tuguni
                    self.knowledge_graph.add_node(section_id, 
                                                type="section", 
                                                bob=section.get('bob', ''), 
                                                title=section_title,
                                                description=section.get('description', ''))
                    self.node_mapping[section_id] = section_id
                    
                    # Bo'limni hisobot shakliga bog'lash
                    self.knowledge_graph.add_edge(report_node_id, section_id, relation="contains_section")
                    
                    # Ustunlar
                    if 'columns' in section:
                        for column in section['columns']:
                            col_id = f"column_{section.get('bob', '')}_{column.get('column', '')}"
                            col_desc = column.get('description', '')
                            
                            # Ustun tuguni
                            self.knowledge_graph.add_node(col_id, 
                                                        type="column", 
                                                        bob=section.get('bob', ''), 
                                                        column=column.get('column', ''),
                                                        description=col_desc)
                            self.node_mapping[col_id] = col_id
                            
                            # Ustunni bo'limga bog'lash
                            self.knowledge_graph.add_edge(section_id, col_id, relation="has_column")
                            
                            # Mantiqiy nazoratlar
                            if 'strict_logical_controls' in column:
                                for i, control in enumerate(column.get('strict_logical_controls', [])):
                                    control_id = f"strict_control_{section.get('bob', '')}_{column.get('column', '')}_{i}"
                                    self.knowledge_graph.add_node(control_id, 
                                                                type="strict_control", 
                                                                description=control)
                                    self.knowledge_graph.add_edge(col_id, control_id, relation="has_strict_control")
                                    self.node_mapping[control_id] = control_id
                            
                            if 'non_strict_logical_controls' in column:
                                for i, control in enumerate(column.get('non_strict_logical_controls', [])):
                                    control_id = f"non_strict_control_{section.get('bob', '')}_{column.get('column', '')}_{i}"
                                    self.knowledge_graph.add_node(control_id, 
                                                                type="non_strict_control", 
                                                                description=control)
                                    self.knowledge_graph.add_edge(col_id, control_id, relation="has_non_strict_control")
                                    self.node_mapping[control_id] = control_id
                    
                    # Qatorlar
                    if 'rows' in section:
                        for row in section['rows']:
                            row_id = f"row_{section.get('bob', '')}_{row.get('row_code', '')}"
                            
                            # Qator tuguni
                            self.knowledge_graph.add_node(row_id, 
                                                        type="row", 
                                                        bob=section.get('bob', ''), 
                                                        row_code=row.get('row_code', ''),
                                                        name=row.get('name', ''))
                            self.node_mapping[row_id] = row_id
                            
                            # Qatorni bo'limga bog'lash
                            self.knowledge_graph.add_edge(section_id, row_id, relation="has_row")
                            
                            # Qator ustunlari
                            if 'columns' in row:
                                for column in row['columns']:
                                    cell_id = f"cell_{section.get('bob', '')}_{row.get('row_code', '')}_{column.get('column', '')}"
                                    col_id = f"column_{section.get('bob', '')}_{column.get('column', '')}"
                                    
                                    # Katakcha tuguni
                                    self.knowledge_graph.add_node(cell_id, 
                                                                type="cell", 
                                                                bob=section.get('bob', ''), 
                                                                row_code=row.get('row_code', ''),
                                                                column=column.get('column', ''),
                                                                description=column.get('description', ''))
                                    self.node_mapping[cell_id] = cell_id
                                    
                                    # Katakchani qatorga bog'lash
                                    self.knowledge_graph.add_edge(row_id, cell_id, relation="has_cell")
                                    
                                    # Katakchani ustunga bog'lash (agar ustun mavjud bo'lsa)
                                    if col_id in self.node_mapping:
                                        self.knowledge_graph.add_edge(col_id, cell_id, relation="cell_in_column")
        
        logging.info(f"Knowledge Graph yaratildi: {len(self.knowledge_graph.nodes)} tugun, {len(self.knowledge_graph.edges)} bog'lanish")

    def _graph_search(self, so_rov: str, return_raw: bool = False) -> List[Dict]:
        """Knowledge Graph orqali qidiruv."""
        if not self.use_knowledge_graph or not self.knowledge_graph:
            logging.warning("Knowledge Graph ishlatilmayapti yoki yaratilmagan")
            return []
        
        logging.info(f"Knowledge Graph qidiruvini boshlash: {so_rov}")
        results = []
        
        # So'rovdagi muhim so'zlarni ajratib olish
        important_words = self._extract_important_words(so_rov)
        
        # Tugunlar bo'yicha qidirish
        for node_id, node_data in self.knowledge_graph.nodes(data=True):
            node_text = ""
            match_score = 0
            
            # Tugun ma'lumotlarini matn ko'rinishiga o'tkazish
            for key, value in node_data.items():
                if isinstance(value, str):
                    node_text += f"{value} "
            
            # Muhim so'zlar bo'yicha moslikni tekshirish
            for word in important_words:
                if word.lower() in node_text.lower():
                    match_score += 1
            
            # Agar moslik topilgan bo'lsa
            if match_score > 0:
                node_type = node_data.get('type', '')
                
                # Tugun turiga qarab natijani shakllantirish
                if node_type == 'section':
                    results.append({
                        'turi': 'bo\'lim',
                        'nomi': node_data.get('title', ''),
                        'bob': node_data.get('bob', ''),
                        'tavsifi': node_data.get('description', ''),
                        'bo\'lim_turi': '',
                        'o\'xshashlik': match_score / len(important_words) if important_words else 0
                    })
                elif node_type == 'column':
                    results.append({
                        'turi': 'ustun',
                        'nomi': node_data.get('description', ''),
                        'kodi': node_data.get('column', ''),
                        'bob': node_data.get('bob', ''),
                        'tavsifi': node_data.get('description', ''),
                        'bo\'lim_turi': '',
                        'o\'xshashlik': match_score / len(important_words) if important_words else 0
                    })
                elif node_type == 'row':
                    results.append({
                        'turi': 'qator',
                        'nomi': node_data.get('name', ''),
                        'qator_kodi': node_data.get('row_code', ''),
                        'bob': node_data.get('bob', ''),
                        'tavsifi': node_data.get('name', ''),
                        'bo\'lim_turi': '',
                        'o\'xshashlik': match_score / len(important_words) if important_words else 0
                    })
                elif node_type in ['strict_control', 'non_strict_control']:
                    # Nazorat tugunining bog'lanishlarini tekshirish
                    for pred in self.knowledge_graph.predecessors(node_id):
                        pred_data = self.knowledge_graph.nodes[pred]
                        if pred_data.get('type') == 'column':
                            results.append({
                                'turi': 'nazorat',
                                'nomi': node_data.get('description', ''),
                                'kodi': pred_data.get('column', ''),
                                'bob': pred_data.get('bob', ''),
                                'tavsifi': node_data.get('description', ''),
                                'nazorat_turi': 'qatiy' if node_type == 'strict_control' else 'qatiy_emas',
                                'bo\'lim_turi': '',
                                'o\'xshashlik': match_score / len(important_words) if important_words else 0
                            })
        
        # Bog'lanishlar orqali qo'shimcha ma'lumotlarni qidirish
        # Masalan, agar qator topilgan bo'lsa, uning barcha katakchalari va bog'liq ustunlarini ham qaytarish
        enhanced_results = []
        for result in results:
            enhanced_results.append(result)
            
            if result.get('turi') == 'qator':
                bob = result.get('bob', '')
                row_code = result.get('qator_kodi', '')
                row_id = f"row_{bob}_{row_code}"
                
                # Qatorning barcha katakchalarini topish
                if row_id in self.node_mapping:
                    for succ in self.knowledge_graph.successors(row_id):
                        succ_data = self.knowledge_graph.nodes[succ]
                        if succ_data.get('type') == 'cell':
                            # Katakcha ma'lumotlarini qo'shish
                            cell_result = {
                                'turi': 'katakcha',
                                'qator_kodi': row_code,
                                'ustun': succ_data.get('column', ''),
                                'bob': bob,
                                'tavsifi': succ_data.get('description', ''),
                                'bo\'lim_turi': '',
                                'o\'xshashlik': result.get('o\'xshashlik', 0) * 0.9  # Biroz kamroq o'xshashlik
                            }
                            enhanced_results.append(cell_result)
        
        # Natijalarni o'xshashlik bo'yicha tartiblash
        enhanced_results.sort(key=lambda x: x.get('o\'xshashlik', 0), reverse=True)
        
        logging.info(f"Knowledge Graph qidiruvida {len(enhanced_results)} natija topildi")
        return enhanced_results if return_raw else self._format_results(enhanced_results)

    def _get_logical_controls(self, bob: str, ustun: str) -> Dict:
        """Berilgan bob va ustun uchun mantiqiy nazoratlarni qaytaradi."""
        nazoratlar = {
            'qatiy_nazoratlar': [],
            'qatiy_bolmagan_nazoratlar': []
        }
        
        # Bobni topish
        for item in self.invest12_data:
            if 'sections' in item:
                for section in item['sections']:
                    if section.get('bob', '') == bob:
                        # Ustunni topish
                        # Avval section.columns da qidirish (1-BOB kabi)
                        if 'columns' in section:
                            for column in section['columns']:
                                if column.get('column', '') == ustun:
                                    nazoratlar['qatiy_nazoratlar'] = column.get('strict_logical_controls', [])
                                    nazoratlar['qatiy_bolmagan_nazoratlar'] = column.get('non_strict_logical_controls', [])
                                    return nazoratlar
                        
                        # Keyin rows.columns da qidirish
                        if 'rows' in section:
                            for row in section['rows']:
                                if 'columns' in row:
                                    for column in row['columns']:
                                        if column.get('column', '') == ustun:
                                            nazoratlar['qatiy_nazoratlar'] = column.get('strict_logical_controls', [])
                                            nazoratlar['qatiy_bolmagan_nazoratlar'] = column.get('non_strict_logical_controls', [])
                                            return nazoratlar
        
        return nazoratlar

    def _format_results(self, results: List[Dict]) -> Dict:
        """Qidiruv natijalarini formatlash."""
        if not results:
            return {"holat": "natijalar_yoq", "soni": 0, "natijalar": []}
        
        formatted_results = []
        for result in results:
            formatted_result = {
                'nomi': result.get('nomi', ''),
                'kodi': result.get('kodi', ''),
                'tavsifi': result.get('tavsifi', ''),
                'bo\'lim_turi': result.get('bo\'lim_turi', ''),
                'bob': result.get('bob', '')
            }
            
            if 'ustunlar' in result:
                formatted_result['ustunlar'] = result['ustunlar']
            if 'o\'xshashlik' in result:
                formatted_result['o\'xshashlik'] = round(result['o\'xshashlik'], 3)
            if 'tur' in result:
                formatted_result['tur'] = result['tur']
            if 'ustun' in result:
                formatted_result['ustun'] = result['ustun']
            if 'qator_kodi' in result:
                formatted_result['qator_kodi'] = result['qator_kodi']
            
            # Mantiqiy nazoratlarni qo'shish
            if result.get('tur') == 'ustun' or result.get('tur') == 'ustun_nom':
                # Invest12 JSON faylidan ustun uchun mantiqiy nazoratlarni topish
                bob = result.get('bob', '')
                ustun = result.get('kodi', '')
                
                if bob and ustun:
                    nazoratlar = self._get_logical_controls(bob, ustun)
                    if nazoratlar:
                        formatted_result['mantiqiy_nazoratlar'] = nazoratlar
        
            formatted_results.append(formatted_result)
    
        return {
            "holat": "muvaffaqiyatli",
            "soni": len(formatted_results),
            "natijalar": formatted_results
        }

    def _run(self, so_rov: str) -> Dict:
        """Toolni ishga tushirish."""
        logging.info(f"Qidiruv so'rovi: {so_rov}")
        
        # Knowledge Graph qidiruvini amalga oshirish (agar yoqilgan bo'lsa)
        graph_results = []
        if self.use_knowledge_graph and self.knowledge_graph:
            logging.info("Knowledge Graph qidiruvini boshlash")
            graph_results = self._graph_search(so_rov, return_raw=True)
        
        # Matn qidiruvini amalga oshirish
        text_results = self._search_by_text(so_rov, return_raw=True)
        
        # Agar matn qidiruvida natija topilmagan bo'lsa yoki natijalar soni kam bo'lsa
        if (not text_results or len(text_results) < 3) and self.use_embeddings:
            logging.info("Semantik qidiruv ishga tushirilmoqda")
            semantic_results = self._semantic_search(so_rov, return_raw=True)
            
            # Natijalarni birlashtirish
            if semantic_results:
                # Takrorlanishlarni oldini olish uchun
                existing_items = set()
                if text_results:
                    for item in text_results:
                        item_key = f"{item.get('bob', '')}-{item.get('kodi', '')}-{item.get('ustun', '')}"
                        existing_items.add(item_key)
                
                # Semantik qidiruv natijalarini qo'shish
                for item in semantic_results:
                    item_key = f"{item.get('bob', '')}-{item.get('kodi', '')}-{item.get('ustun', '')}"
                    if item_key not in existing_items:
                        text_results.append(item)
        
        # Knowledge Graph natijalarini qo'shish
        if graph_results:
            # Takrorlanishlarni oldini olish uchun
            existing_items = set()
            if text_results:
                for item in text_results:
                    item_key = f"{item.get('bob', '')}-{item.get('kodi', '')}-{item.get('ustun', '')}"
                    existing_items.add(item_key)
            
            # Knowledge Graph natijalarini qo'shish
            for item in graph_results:
                item_key = f"{item.get('bob', '')}-{item.get('kodi', '')}-{item.get('ustun', '')}"
                if item_key not in existing_items:
                    text_results.append(item)
        
        # Agar natijalar bo'sh bo'lsa va embedding ishlatilmasa, semantik qidiruvni majburiy ishlatish
        if not text_results and not self.use_embeddings:
            logging.info("Matn qidiruvida natija topilmadi, semantik qidiruv majburiy ishga tushirilmoqda")
            return self._semantic_search(so_rov)
        
        return self._format_results(text_results)