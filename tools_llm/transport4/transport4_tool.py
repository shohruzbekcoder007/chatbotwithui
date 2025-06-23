import json
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, List
from retriever.langchain_chroma import CustomEmbeddingFunction
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Transport4ToolInput(BaseModel):
    query: str = Field(description="4-transport shakli hisoboti ma'lumotlarini qidirish uchun so'rov. Bu yerga 4-transport shakli hisoboti ma'lumotlarini qidirish uchun so'rovni kiriting.")

class Transport4Tool(BaseTool):
    """Transport ma'lumotlarini qidirish uchun tool."""
    
    name: str = "transport4_tool"
    description: str = "4-transport shakli hisoboti ma'lumotlarini qidirish va tahlil qilish uchun tool."

    args_schema: Type[BaseModel] = Transport4ToolInput

    embedding_model: Optional[CustomEmbeddingFunction] = Field(default=None)
    transport4_data: Dict = Field(default_factory=dict)
    entity_texts: List[str] = Field(default_factory=list)
    entity_infos: List[Dict] = Field(default_factory=list)
    use_embeddings: bool = Field(default=True)
    entity_embeddings_cache: Optional[torch.Tensor] = Field(default=None)

    def __init__(self, transport4_file_path: str, use_embeddings: bool = True, embedding_model=None):
        """Initialize the Transport4 tool with the path to the Transport4 JSON file."""
        super().__init__()
        self.transport4_data = self._load_transport4_data(transport4_file_path)
        self.use_embeddings = use_embeddings
        
        # Embedding modelini tashqaridan olish
        if embedding_model is not None:
            self.embedding_model = embedding_model
            self._prepare_embedding_data()
        else:
            # Lazy loading - embedding modeli kerak bo'lganda yuklanadi
            self.embedding_model = None

    def _load_transport4_data(self, transport4_file_path: str) -> Dict:
        """Transport4 ma'lumotlarini yuklash"""
        with open(transport4_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _prepare_embedding_data(self):
        """Transport4 ma'lumotlarini embedding uchun tayyorlash."""
        self.entity_texts = []
        self.entity_infos = []
        
        # Hisobot shakli umumiy ma'lumotlarini qo'shish
        for item in self.transport4_data:
            if 'report_form' in item:
                report_form = item['report_form']
                # Hisobot shakli umumiy ma'lumotlarini qo'shish
                text = f"4-transport shakli: {report_form.get('name', '')}. {report_form.get('submitters', '')}. {report_form.get('structure', '')}. {report_form.get('indicators', '')}"
                self.entity_texts.append(text)
                self.entity_infos.append({
                    'type': 'report_form',
                    'content': report_form,
                    'section_type': report_form.get('section_type', '')
                })
                
            # Bo'limlar ma'lumotlarini qo'shish
            if 'sections' in item:
                for section in item['sections']:
                    # Bo'lim umumiy ma'lumotlarini qo'shish
                    section_text = f"{section.get('bob', '')}. {section.get('title', '')}. {section.get('description', '')}"
                    self.entity_texts.append(section_text)
                    self.entity_infos.append({
                        'type': 'section',
                        'content': {
                            'bob': section.get('bob', ''),
                            'title': section.get('title', ''),
                            'description': section.get('description', '')
                        },
                        'section_type': section.get('section_type', '')
                    })
                    
                    # Qatorlar ma'lumotlarini qo'shish
                    if 'rows' in section:
                        for row in section['rows']:
                            row_code = row.get('row_code', '')
                            
                            # Har bir ustun ma'lumotlarini qo'shish
                            if 'columns' in row:
                                for column in row['columns']:
                                    col_num = column.get('column', '')
                                    description = column.get('description', '')
                                    
                                    # Ustun ma'lumotlarini qo'shish
                                    col_text = f"{section.get('bob', '')} {row_code}-satr {col_num}-ustun: {description}"
                                    self.entity_texts.append(col_text)
                                    self.entity_infos.append({
                                        'type': 'column',
                                        'content': {
                                            'bob': section.get('bob', ''),
                                            'row_code': row_code,
                                            'column': col_num,
                                            'description': description
                                        },
                                        'section_type': section.get('section_type', '')
                                    })
                                    
                                    # Mantiqiy nazoratlar ma'lumotlarini qo'shish
                                    strict_controls = column.get('strict_logical_controls', [])
                                    non_strict_controls = column.get('non_strict_logical_controls', [])
                                    
                                    for control in strict_controls:
                                        control_text = f"{section.get('bob', '')} {row_code}-satr {col_num}-ustun qat'iy mantiqiy nazorat: {control}"
                                        self.entity_texts.append(control_text)
                                        self.entity_infos.append({
                                            'type': 'strict_control',
                                            'content': {
                                                'bob': section.get('bob', ''),
                                                'row_code': row_code,
                                                'column': col_num,
                                                'control': control
                                            },
                                            'section_type': section.get('section_type', '')
                                        })
                                    
                                    for control in non_strict_controls:
                                        control_text = f"{section.get('bob', '')} {row_code}-satr {col_num}-ustun qat'iy bo'lmagan mantiqiy nazorat: {control}"
                                        self.entity_texts.append(control_text)
                                        self.entity_infos.append({
                                            'type': 'non_strict_control',
                                            'content': {
                                                'bob': section.get('bob', ''),
                                                'row_code': row_code,
                                                'column': col_num,
                                                'control': control
                                            },
                                            'section_type': section.get('section_type', '')
                                        })
        
        # Embedding keshini tozalash
        self.entity_embeddings_cache = None

    def _get_general_info(self) -> str:
        """Get general information about the transport4 data."""
        return """
        Transport4 ma'lumotlari:
        
        """

    def _extract_important_words(self, query: str) -> List[str]:
        """So'rovdan muhim so'zlarni ajratib olish"""
        # So'zlarni ajratib olish
        words = query.lower().split()
        
        # Stop so'zlarni o'chirish
        stop_words = ["va", "bilan", "uchun", "ham", "lekin", "ammo", "yoki", "agar", "chunki", "shuning", "bo'yicha", "haqida", "qanday", "nima", "qaysi", "nechta", "qanaqa", "qachon", "qayerda", "kim", "nima", "necha"]
        
        # Muhim so'zlarni ajratib olish - uzunligi 3 dan katta va stop so'z bo'lmagan so'zlar
        important_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return important_words

    def _search_by_text(self, query: str, return_raw: bool = False) -> Optional[Dict]:
        """Search for a specific text in the transport4 data."""
        # Muhim so'zlarni ajratib olish
        important_words = self._extract_important_words(query)
        logging.info(f"Matn qidiruv uchun muhim so'zlar: {important_words}")
        
        # Natijalarni saqlash uchun ro'yxat
        results = []

        for entity in self.transport4_data:
            if entity['NAME'].lower() == query.lower() or entity['CODE'] == query:
                results.append(entity)

        # Natijalarni qaytarish
        if return_raw:
            return results
        else:
            return self._format_results(results)

    # shu functionni o'zgartirish kerak
    def _format_results(self, results: List[Dict]) -> Dict:
        """Format the search results."""
        formatted_results = []
        for result in results:
            formatted_result = {
                'name': result['NAME'],
                'code': result['CODE'],
                'description': result['DESCRIPTION']
            }
            formatted_results.append(formatted_result)
        return formatted_results