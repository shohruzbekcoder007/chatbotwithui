import json
import logging
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional, Dict, List, Union
from retriever.langchain_chroma import CustomEmbeddingFunction
import torch
import torch.nn.functional as F

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
        """
        Matn bo'yicha qidirish
        
        Args:
            query: Qidiruv so'rovi
            return_raw: Xom natijalarni qaytarish kerakmi
            
        Returns:
            Qidiruv natijalari
        """
        important_words = self._extract_important_words(query)
        logging.info(f"Matn qidiruv uchun muhim so'zlar: {important_words}")
        results = []
        
        # So'rovni qismlarga ajratish
        query_parts = query.lower().split()
        row_part = None
        column_part = None
        
        # Qator va ustun qismlarini aniqlash
        for part in query_parts:
            if 'satr' in part or 'qator' in part or part.isdigit():
                # Raqamni ajratib olish
                row_digits = ''.join(filter(str.isdigit, part))
                if row_digits:
                    row_part = row_digits
            elif 'ustun' in part or 'column' in part:
                # Raqamni ajratib olish
                column_digits = ''.join(filter(str.isdigit, part))
                if column_digits:
                    column_part = column_digits
        
        # Transport4 ma'lumotlarini qidirish
        for item in self.transport4_data:
            # Report form qismini tekshirish
            if 'report_form' in item:
                report_form = item['report_form']
                if query.lower() in report_form.get('name', '').lower() or query.lower() in report_form.get('submitters', '').lower():
                    results.append({
                        'NAME': report_form.get('name', ''),
                        'CODE': '',
                        'DESCRIPTION': report_form.get('submitters', ''),
                        'SECTION_TYPE': 'report_form',
                        'STRUCTURE': report_form.get('structure', ''),
                        'INDICATORS': report_form.get('indicators', '')
                    })
            
            # Sections qismini tekshirish
            if 'sections' in item:
                for section in item['sections']:
                    bob = section.get('bob', '')
                    title = section.get('title', '')
                    description = section.get('description', '')
                    
                    if query.lower() in bob.lower() or query.lower() in title.lower():
                        results.append({
                            'NAME': bob,
                            'CODE': '',
                            'DESCRIPTION': description,
                            'SECTION_TYPE': section.get('section_type', ''),
                            'BOB': bob,
                            'TITLE': title
                        })
                    
                    # Rows qismini tekshirish
                    if 'rows' in section:
                        for row in section['rows']:
                            row_code = row.get('row_code', '')
                            
                            # Qator kodi so'rovga mos kelsa yoki qator qismi aniqlangan bo'lsa
                            if (query.lower() in row_code.lower() or 
                                query == row_code or 
                                (row_part and row_part == row_code)):
                                
                                row_info = {
                                    'NAME': f"{bob} - {row_code}",
                                    'CODE': row_code,
                                    'DESCRIPTION': '',
                                    'SECTION_TYPE': section.get('section_type', ''),
                                    'BOB': bob,
                                    'COLUMNS': []
                                }
                                
                                # Columns qismini tekshirish
                                if 'columns' in row:
                                    for column in row['columns']:
                                        column_num = column.get('column', '')
                                        column_desc = column.get('description', '')
                                        
                                        # Agar ustun qismi aniqlangan bo'lsa va mos kelsa
                                        if column_part and column_part == column_num:
                                            row_info = {
                                                'NAME': f"{bob} - {row_code} - Ustun {column_num}",
                                                'CODE': f"{row_code}-{column_num}",
                                                'DESCRIPTION': column_desc,
                                                'SECTION_TYPE': section.get('section_type', ''),
                                                'BOB': bob,
                                                'ROW_CODE': row_code,
                                                'COLUMN': column_num
                                            }
                                            results.append(row_info)
                                            continue
                                        
                                        row_info['DESCRIPTION'] += column_desc + ' '
                                        # Ensure 'COLUMNS' key exists before appending to it
                                        if 'COLUMNS' not in row_info:
                                            row_info['COLUMNS'] = []
                                        row_info['COLUMNS'].append({
                                            'column': column_num,
                                            'description': column_desc
                                        })
                                
                                # Agar ustun qismi aniqlanmagan bo'lsa, butun qator ma'lumotlarini qo'shish
                                if not column_part:
                                    results.append(row_info)
                            
                            # Logical controls qismini tekshirish
                            if 'logical_controls' in row:
                                for control in row['logical_controls']:
                                    control_code = control.get('control_code', '')
                                    control_desc = control.get('description', '')
                                    
                                    if query == control_code or query.lower() in control_code.lower():
                                        results.append({
                                            'NAME': f"{bob} - {row_code} - {control_code}",
                                            'CODE': control_code,
                                            'DESCRIPTION': control_desc,
                                            'SECTION_TYPE': 'logical_control',
                                            'BOB': bob,
                                            'ROW_CODE': row_code
                                        })

        # Natijalarni qaytarish
        if return_raw:
            return results
        else:
            return self._format_results(results)

    def _semantic_search(self, query: str, return_raw: bool = False) -> Union[List[Dict], str]:
        """Semantik qidiruv - embedding modelini ishlatib o'xshash ma'lumotlarni topish"""
        try: 
            if not self.entity_texts or not self.entity_infos:
                logging.info("Semantik qidiruv: Ma'lumotlar mavjud emas.")
                return [] if return_raw else "Ma'lumotlar mavjud emas."
            
            if not self.use_embeddings:
                logging.info("Semantik qidiruv: Semantik qidiruv o'chirilgan.")
                return [] if return_raw else "Semantik qidiruv o'chirilgan."

            # Agar embedding model yuklanmagan bo'lsa
            if self.embedding_model is None:
                logging.info("Embedding modeli yuklanmoqda...")
                try:
                    from retriever.langchain_chroma import CustomEmbeddingFunction
                    self.embedding_model = CustomEmbeddingFunction()
                    self._prepare_embedding_data()
                except Exception as e:
                    logging.error(f"Embedding modelini yuklashda xatolik: {str(e)}")
                    return [] if return_raw else f"Embedding modelini yuklashda xatolik: {str(e)}"

            # So'rovni tozalash va tayyorlash
            query_lower = query.lower()
            
            # Maxsus so'zlarni almashtirish
            if "stir" in query_lower:
                query_lower = query_lower.replace("stir", "").strip()
                logging.info(f"Semantik qidiruvda STIR so'rovi INN sifatida qidirilmoqda: {query_lower}")
            
            if "ktut" in query_lower:
                query_lower = query_lower.replace("ktut", "").strip()
                logging.info(f"Semantik qidiruvda KTUT so'rovi OKPO sifatida qidirilmoqda: {query_lower}")

            # So'rov embeddingini olish - embed_query o'rniga encode metodini ishlatish
            logging.info("So'rov embeddingini olish...")
            try:
                # encode metodi listni kutadi, shuning uchun [query_lower] ko'rinishida beramiz
                query_embedding = self.embedding_model.encode([query_lower], normalize=True)[0]
                logging.info("So'rov embeddingini olish muvaffaqiyatli yakunlandi")
            except Exception as e:
                logging.error(f"So'rov embeddingini olishda xatolik: {str(e)}")
                return [] if return_raw else f"So'rov embeddingini olishda xatolik: {str(e)}"
            
            # Agar entity embeddinglari keshlanmagan bo'lsa, hisoblash
            if self.entity_embeddings_cache is None:
                logging.info("Entity embeddinglari hisoblanmoqda...")
                try:
                    # encode metodi listni kutadi
                    self.entity_embeddings_cache = self.embedding_model.encode(self.entity_texts)
                    logging.info(f"Entity embeddinglari hisoblandi: {len(self.entity_embeddings_cache)} ta")
                except Exception as e:
                    logging.error(f"Entity embeddinglarini hisoblashda xatolik: {str(e)}")
                    return [] if return_raw else f"Entity embeddinglarini hisoblashda xatolik: {str(e)}"
            
            query_embedding_tensor = torch.tensor(query_embedding)
            entity_embeddings_tensor = torch.tensor(self.entity_embeddings_cache)
            
            # Cosine o'xshashlikni hisoblash
            similarities = F.cosine_similarity(
                query_embedding_tensor.unsqueeze(0), 
                entity_embeddings_tensor, 
                dim=1
            )
            
            # Eng o'xshash 5ta natijani topish
            top_k = 5
            top_indices = torch.argsort(similarities, descending=True)[:top_k]
            
            # Natijalarni tayyorlash
            results = []
            logging.info(f"Top {top_k} natijalar:")
            for idx in top_indices:
                similarity = float(similarities[idx])
                logging.info(f"Indeks: {idx}, O'xshashlik: {similarity:.4f}, Matn: {self.entity_texts[idx][:100]}...")
                
                if similarity > 0.5:  # Minimal o'xshashlik chegarasi
                    entity_info = self.entity_infos[idx]
                    results.append({
                        'NAME': entity_info.get('content', {}).get('title', '') or 
                               entity_info.get('content', {}).get('description', ''),
                        'CODE': entity_info.get('content', {}).get('row_code', '') or 
                               entity_info.get('content', {}).get('column', ''),
                        'DESCRIPTION': self.entity_texts[idx],
                        'SECTION_TYPE': entity_info.get('section_type', ''),
                        'BOB': entity_info.get('content', {}).get('bob', ''),
                        'TYPE': entity_info.get('type', ''),
                        'SIMILARITY': similarity
                    })
            
            # Natijalarni qaytarish
            if return_raw:
                return results
            else:
                if results:
                    return self._format_results(results)
                else:
                    return {"status": "no_results", "count": 0, "results": []}
                
        except Exception as e:
            logging.error(f"Semantic search error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return [] if return_raw else {"status": "error", "message": str(e), "results": []}

    def _format_results(self, results: List[Dict]) -> Dict:
        """Format the search results.
        
        Args:
            results: List of search results from transport4 data
            
        Returns:
            Dict containing formatted results and metadata
        """
        if not results:
            return {"status": "natijalar_yoq", "count": 0, "results": []}
        
        formatted_results = []
        for result in results:
            # Basic information extraction
            formatted_result = {
                'NAME': result.get('NAME', ''),
                'CODE': result.get('CODE', ''),
                'DESCRIPTION': result.get('DESCRIPTION', '')
            }
            
            # Add additional information if available
            if 'SECTION_TYPE' in result:
                # O'zbek tiliga o'girish
                section_type = result['SECTION_TYPE']
                if section_type.lower() == 'statistic':
                    formatted_result['bo_lim_turi'] = 'statistika'
                else:
                    formatted_result['bo_lim_turi'] = section_type
        
            if 'BOB' in result:
                formatted_result['bob'] = result['BOB']
            
            if 'COLUMNS' in result and isinstance(result['COLUMNS'], list):
                formatted_result['ustunlar'] = [
                    {
                        'ustun_raqami': col.get('column', ''),
                        'tavsifi': col.get('description', '')
                    }
                    for col in result['COLUMNS']
                ]
        
            # Add any logical controls if present
            if 'STRICT_CONTROLS' in result and result['STRICT_CONTROLS']:
                formatted_result['qat_iy_nazoratlar'] = result['STRICT_CONTROLS']
            
            if 'NON_STRICT_CONTROLS' in result and result['NON_STRICT_CONTROLS']:
                formatted_result['qat_iy_bo_lmagan_nazoratlar'] = result['NON_STRICT_CONTROLS']
        
            # Semantik qidiruv natijalarini formatlash
            if 'SIMILARITY' in result:
                formatted_result['o_hshashlik'] = round(result['SIMILARITY'], 3)
                formatted_result['tur'] = result.get('TYPE', '')
        
            if 'COLUMN' in result:
                formatted_result['ustun'] = result['COLUMN']
            
            if 'ROW_CODE' in result:
                formatted_result['qator_kodi'] = result['ROW_CODE']
        
            formatted_results.append(formatted_result)
    
        # Return a structured response with metadata
        return {
            "status": "muaffaqiyatli",
            "count": len(formatted_results),
            "results": formatted_results
        }

    def _run(self, query: str) -> Dict:
        """Run the tool with the given query."""
        logging.info(f"Transport4Tool qidiruv so'rovi: {query}")
        
        # Matn qidiruvini bajarish
        text_results = self._search_by_text(query, return_raw=True)
        logging.info(f"Matn qidiruvida {len(text_results)} natija topildi")
        
        # Debug: print text search results
        if text_results:
            logging.info("Matn qidiruvida topilgan natijalar:")
            for i, result in enumerate(text_results[:3]):  # Show only first 3 results to avoid log clutter
                logging.info(f"  {i+1}. {result.get('NAME', 'No Name')} - {result.get('DESCRIPTION', 'No Description')[:100]}...")
        
        # Agar matn qidiruvida natijalar topilmagan bo'lsa, semantik qidiruvni sinab ko'rish
        if not text_results and self.use_embeddings:
            logging.info("Matn qidiruvida natija topilmadi, semantik qidiruv ishga tushirilmoqda")
            semantic_results = self._semantic_search(query, return_raw=True)
            
            if isinstance(semantic_results, list):
                logging.info(f"Semantik qidiruvda {len(semantic_results)} natija topildi")
                
                # Debug: print semantic search results
                if semantic_results:
                    logging.info("Semantik qidiruvda topilgan natijalar:")
                    for i, result in enumerate(semantic_results[:3]):  # Show only first 3 results
                        logging.info(f"  {i+1}. {result.get('NAME', 'No Name')} - {result.get('DESCRIPTION', 'No Description')[:100]}...")
                
                return self._format_results(semantic_results)
            else:
                logging.warning(f"Semantik qidiruv xatolik qaytardi: {semantic_results}")
                return {"status": "xatolik", "message": "Qidiruv natijasiz yakunlandi", "results": []}
        
        # Matn qidiruvida natijalar topilgan bo'lsa
        return self._format_results(text_results)