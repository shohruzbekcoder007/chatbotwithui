# tools_llm/transport4/transport4_tool.py
import json
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, List, Union, Any
import torch
import torch.nn.functional as F

# Import CustomEmbeddingFunction tipini to'g'ri yo'l bilan
from retriever.langchain_chroma import CustomEmbeddingFunction

# Logging sozlamalari (o'zbek tilida)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Transport4ToolInput(BaseModel):
    so_rov: str = Field(description="4-transport shakli hisoboti ma'lumotlarini qidirish uchun so'rov.")

class Transport4Tool(BaseTool):
    """4-transport shakli hisoboti ma'lumotlarini qidirish va tahlil qilish uchun vosita."""
    
    name: str = "transport4_tool"
    description: str = "4-transport shakli hisoboti ma'lumotlarini qidirish va tahlil qilish uchun vosita."
    args_schema: Type[BaseModel] = Transport4ToolInput

    embedding_model: Optional['CustomEmbeddingFunction'] = Field(default=None)
    transport4_data: Dict = Field(default_factory=dict)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)

    def __init__(self, transport4_file_path: str, use_embeddings: bool = True, embedding_model=None):
        super().__init__()
        self.transport4_data = self._load_transport4_data(transport4_file_path)
        self.use_embeddings = use_embeddings
        self.embedding_model = embedding_model
        if embedding_model is not None:
            self._prepare_embedding_data()

    def _load_transport4_data(self, transport4_file_path: str) -> Dict:
        """4-transport ma'lumotlarini yuklash."""
        try:
            with open(transport4_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info("4-transport ma'lumotlari muvaffaqiyatli yuklandi")
            return data
        except Exception as e:
            logging.error(f"Ma'lumotlarni yuklashda xatolik: {str(e)}")
            raise

    def _prepare_embedding_data(self):
        """Ma'lumotlarni embedding uchun tayyorlash."""
        self.entity_texts = []
        self.entity_infos = []
        
        for item in self.transport4_data:
            if 'report_form' in item:
                report_form = item['report_form']
                text = f"4-transport shakli: {report_form.get('name', '')}. {report_form.get('submitters', '')}. {report_form.get('structure', '')}. {report_form.get('indicators', '')}"
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

    def _extract_important_words(self, so_rov: str):
        """So'rovdan muhim so'zlarni ajratib olish."""
        words = so_rov.lower().split()
        # Stop so'zlarni olib tashlash
        stop_words = ['va', 'bilan', 'uchun', 'qanday', 'qanaqa', 'qaysi', 'nima', 'nimalar', 'qayerda', 'qachon', 'kim', 'necha', 'qancha']
        important_words = [word for word in words if word not in stop_words]
        return important_words

    def _search_by_text(self, so_rov: str, return_raw: bool = False) -> Union[List[Dict], Dict]:
        """Matn bo'yicha qidirish."""
        important_words = self._extract_important_words(so_rov)
        logging.info(f"Muhim so'zlar: {important_words}")
        results = []
        
        so_rov_parts = so_rov.lower().split()
        qator_part = None
        ustun_part = None
        bob_part = None
        
        # Bob va ustun qismlarini aniqlash
        for i, part in enumerate(so_rov_parts):
            # Bob qismini aniqlash
            if 'bob' in part:
                # Bob raqamini olish
                bob_digits = ''.join(filter(str.isdigit, part))
                if bob_digits:
                    bob_part = bob_digits
                # Agar bob so'zi alohida kelsa, oldingi so'zni tekshirish
                elif i > 0 and so_rov_parts[i-1].isdigit():
                    bob_part = so_rov_parts[i-1]
        
            # Qator qismini aniqlash
            if 'qator' in part or 'satr' in part or part.isdigit():
                qator_digits = ''.join(filter(str.isdigit, part))
                if qator_digits:
                    qator_part = qator_digits
        
            # Ustun qismini aniqlash
            elif 'ustun' in part or 'column' in part:
                # Agar ustun so'zi alohida kelsa, oldingi so'zni tekshirish
                if i > 0:
                    prev_part = so_rov_parts[i-1]
                    # Agar oldingi so'z raqam yoki harf bo'lsa
                    if prev_part.isdigit():
                        ustun_part = prev_part
                    elif len(prev_part) == 1 and prev_part.isalpha():
                        ustun_part = prev_part.upper()
    
        # Agar ustun_part aniqlanmagan bo'lsa, so'rovda alohida harflarni qidirish
        if not ustun_part:
            for part in so_rov_parts:
                if len(part) == 1 and part.isalpha():
                    ustun_part = part.upper()
                    break
    
        for item in self.transport4_data:
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
                    
                    if so_rov.lower() in bob.lower() or so_rov.lower() in sarlavha.lower() or bob_match:
                        results.append({
                            'nomi': bob,
                            'kodi': '',
                            'tavsifi': tavsifi,
                            'bo\'lim_turi': section.get('section_type', ''),
                            'bob': bob,
                            'sarlavha': sarlavha
                        })
                
                # Dynamic section_type uchun
                if section.get('section_type', '') == 'dynamic' and 'columns' in section:
                    # Bob qismini tekshirish
                    if bob_part and bob_part not in bob:
                        continue
                        
                    for column in section['columns']:
                        ustun_num = column.get('column', '')
                        ustun_tavsifi = column.get('description', '')
                        
                        # Ustun qismini tekshirish
                        ustun_match = False
                        if ustun_part:
                            ustun_match = ustun_part == ustun_num
                        
                        if (so_rov.lower() in ustun_tavsifi.lower() or 
                            ustun_match or
                            (bob_part and ustun_part and bob_part in bob and ustun_part == ustun_num)):
                            
                            column_info = {
                                'nomi': f"{bob} - Ustun {ustun_num}",
                                'kodi': ustun_num,
                                'tavsifi': ustun_tavsifi,
                                'bo\'lim_turi': 'dynamic',
                                'bob': bob,
                                'ustun': ustun_num,
                                'sarlavha': sarlavha
                            }
                            
                            # Qat'iy nazoratlarni qo'shish
                            strict_controls = column.get('strict_logical_controls', [])
                            if strict_controls:
                                column_info['qatiy_nazoratlar'] = strict_controls
                            
                            # Qat'iy bo'lmagan nazoratlarni qo'shish
                            non_strict_controls = column.get('non_strict_logical_controls', [])
                            if non_strict_controls:
                                column_info['qatiy_bolmagan_nazoratlar'] = non_strict_controls
                            
                            results.append(column_info)
                
                # Statistic section_type uchun
                if 'rows' in section:
                    for row in section['rows']:
                        qator_kodi = row.get('row_code', '')
                        if (so_rov.lower() in qator_kodi.lower() or 
                            so_rov == qator_kodi or 
                            (qator_part and qator_part == qator_kodi)):
                            
                            qator_info = {
                                'nomi': f"{bob} - {qator_kodi}",
                                'kodi': qator_kodi,
                                'tavsifi': '',
                                'bo\'lim_turi': section.get('section_type', ''),
                                'bob': bob,
                                'ustunlar': []  # Ustunlar kalit so'zini qo'shamiz
                            }
                            
                            if 'columns' in row:
                                for column in row['columns']:
                                    ustun_num = column.get('column', '')
                                    ustun_tavsifi = column.get('description', '')
                                    
                                    if ustun_part and ustun_part == ustun_num:
                                        qator_info = {
                                            'nomi': f"{bob} - {qator_kodi} - Ustun {ustun_num}",
                                            'kodi': f"{qator_kodi}-{ustun_num}",
                                            'tavsifi': ustun_tavsifi,
                                            'bo\'lim_turi': section.get('section_type', ''),
                                            'bob': bob,
                                            'qator_kodi': qator_kodi,
                                            'ustun': ustun_num,
                                            'ustunlar': []  # Ustunlar kalit so'zini qo'shamiz
                                        }
                                        results.append(qator_info)
                                        continue
                                    
                                    # Ustunlar kalit so'zi mavjudligini tekshiramiz
                                    if 'ustunlar' not in qator_info:
                                        qator_info['ustunlar'] = []
                                        
                                    qator_info['tavsifi'] += ustun_tavsifi + ' '
                                    qator_info['ustunlar'].append({
                                        'ustun_raqami': ustun_num,
                                        'tavsifi': ustun_tavsifi
                                    })
                            
                            if not ustun_part:
                                results.append(qator_info)
    
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
            
            formatted_results.append(formatted_result)
        
        return {
            "holat": "muvaffaqiyatli",
            "soni": len(formatted_results),
            "natijalar": formatted_results
        }

    def _run(self, so_rov: str) -> Dict:
        """Toolni ishga tushirish."""
        logging.info(f"Qidiruv so'rovi: {so_rov}")
        text_results = self._search_by_text(so_rov, return_raw=True)
        
        if not text_results and self.use_embeddings:
            logging.info("Matn qidiruvida natija topilmadi, semantik qidiruv ishga tushirilmoqda")
            return self._semantic_search(so_rov)
        
        return self._format_results(text_results)