#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
12-invest hisoboti ma'lumotlarini XLSX fayldan o'qib, LLM uchun tayyorlash va Knowledge Graph yaratish.
"""

import os
import json
import logging
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Logging sozlamalari
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Invest12KnowledgeGraph:
    """12-invest hisoboti ma'lumotlarini XLSX fayldan o'qib, Knowledge Graph yaratish uchun klass."""
    
    def __init__(self, xlsx_file_path: str):
        """
        Invest12KnowledgeGraph klassini yaratish.
        
        Args:
            xlsx_file_path: XLSX fayl yo'li
        """
        self.xlsx_file_path = xlsx_file_path
        self.data = {}
        self.knowledge_graph = nx.DiGraph()
        self.node_mapping = {}
        
    def read_xlsx(self) -> Dict:
        """
        XLSX fayldan ma'lumotlarni o'qish.
        
        Returns:
            Dict: O'qilgan ma'lumotlar
        """
        logger.info(f"XLSX fayldan ma'lumotlarni o'qish: {self.xlsx_file_path}")
        
        try:
            # Excel faylni o'qish
            xls = pd.ExcelFile(self.xlsx_file_path)
            
            # Barcha varaqlarni o'qish
            sheets_data = {}
            for sheet_name in xls.sheet_names:
                logger.info(f"'{sheet_name}' varaqni o'qish")
                df = pd.read_excel(xls, sheet_name)
                sheets_data[sheet_name] = df
            
            # Ma'lumotlarni strukturalash
            self.data = self._structure_data(sheets_data)
            
            return self.data
            
        except Exception as e:
            logger.error(f"XLSX faylni o'qishda xatolik: {str(e)}")
            raise
    
    def _structure_data(self, sheets_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Excel varaqlaridan olingan ma'lumotlarni strukturalash.
        
        Args:
            sheets_data: Excel varaqlaridan olingan ma'lumotlar
            
        Returns:
            Dict: Strukturalangan ma'lumotlar
        """
        structured_data = {
            "report_form": {
                "name": "12-invest shakli",
                "submitters": "Korxona va tashkilotlar",
                "structure": "Choraklik, yillik",
                "indicators": "Investitsiyalar va qurilish faoliyati"
            },
            "sections": []
        }
        
        # Har bir varaqni alohida bo'lim sifatida qo'shish
        for sheet_name, df in sheets_data.items():
            # Bo'sh qatorlarni olib tashlash
            df = df.dropna(how='all')
            
            # Bo'lim ma'lumotlarini olish
            section_info = self._extract_section_info(sheet_name, df)
            
            # Bo'limni qo'shish
            structured_data["sections"].append(section_info)
        
        return structured_data
    
    def _extract_section_info(self, sheet_name: str, df: pd.DataFrame) -> Dict:
        """
        Excel varaqdan bo'lim ma'lumotlarini ajratib olish.
        
        Args:
            sheet_name: Varaq nomi
            df: Varaq ma'lumotlari
            
        Returns:
            Dict: Bo'lim ma'lumotlari
        """
        # Bo'lim turi va raqamini aniqlash
        section_type = "dynamic"  # Default qiymat
        bob = "1"  # Default qiymat
        
        # Varaq nomidan bo'lim raqamini olish
        if "bob" in sheet_name.lower():
            parts = sheet_name.split()
            for part in parts:
                if part.isdigit():
                    bob = part
                    break
        
        # Sarlavhani aniqlash
        title = sheet_name
        description = f"{bob}-BOB. {title}"
        
        # Bo'lim ma'lumotlarini yaratish
        section_info = {
            "bob": bob,
            "title": title,
            "description": description,
            "section_type": section_type,
            "columns": [],
            "rows": []
        }
        
        # Ustunlarni aniqlash
        columns = self._extract_columns(df)
        section_info["columns"] = columns
        
        # Qatorlarni aniqlash
        rows = self._extract_rows(df, columns)
        section_info["rows"] = rows
        
        return section_info
    
    def _extract_columns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Excel varaqdan ustunlarni ajratib olish.
        
        Args:
            df: Varaq ma'lumotlari
            
        Returns:
            List[Dict]: Ustunlar ro'yxati
        """
        columns = []
        
        # Birinchi qatorni ustun nomlari sifatida olish
        header_row = df.iloc[0]
        
        # Ustunlarni yaratish
        for i, col_name in enumerate(df.columns[1:], 1):  # Birinchi ustunni tashlab ketish (qator kodi)
            column_letter = chr(65 + i)  # A, B, C, ...
            
            # Ustun tavsifi
            description = str(header_row[col_name]) if pd.notna(header_row[col_name]) else f"{column_letter}-ustun"
            
            # Ustun ma'lumotlarini yaratish
            column_info = {
                "column": column_letter,
                "description": description,
                "strict_logical_controls": [],
                "non_strict_logical_controls": []
            }
            
            columns.append(column_info)
        
        return columns
    
    def _extract_rows(self, df: pd.DataFrame, columns: List[Dict]) -> List[Dict]:
        """
        Excel varaqdan qatorlarni ajratib olish.
        
        Args:
            df: Varaq ma'lumotlari
            columns: Ustunlar ro'yxati
            
        Returns:
            List[Dict]: Qatorlar ro'yxati
        """
        rows = []
        
        # Birinchi qatorni tashlab ketish (ustun nomlari)
        for i, row in df.iloc[1:].iterrows():
            # Qator kodi va nomi
            row_code = str(row.iloc[0]) if pd.notna(row.iloc[0]) else str(i)
            row_name = row_code
            
            # Qator ma'lumotlarini yaratish
            row_info = {
                "row_code": row_code,
                "name": row_name,
                "columns": []
            }
            
            # Qator ustunlarini yaratish
            for j, column in enumerate(columns):
                col_idx = j + 1  # Birinchi ustunni tashlab ketish (qator kodi)
                
                # Ustun qiymati
                value = str(row.iloc[col_idx]) if pd.notna(row.iloc[col_idx]) else ""
                
                # Ustun ma'lumotlarini yaratish
                column_info = {
                    "column": column["column"],
                    "description": value
                }
                
                row_info["columns"].append(column_info)
            
            rows.append(row_info)
        
        return rows
    
    def build_knowledge_graph(self) -> nx.DiGraph:
        """
        Ma'lumotlar asosida Knowledge Graph yaratish.
        
        Returns:
            nx.DiGraph: Yaratilgan Knowledge Graph
        """
        logger.info("Knowledge Graph yaratilmoqda...")
        
        # Agar ma'lumotlar mavjud bo'lmasa, XLSX faylni o'qish
        if not self.data:
            self.read_xlsx()
        
        # NetworkX kutubxonasi yordamida yo'naltirilgan graf yaratish
        self.knowledge_graph = nx.DiGraph()
        self.node_mapping = {}
        
        # Hisobot shakli uchun asosiy tugun
        report_node_id = "report_form"
        self.knowledge_graph.add_node(report_node_id, type="report_form", name="12-invest shakli")
        self.node_mapping[report_node_id] = report_node_id
        
        # Hisobot shakli ma'lumotlari
        if "report_form" in self.data:
            report_form = self.data["report_form"]
            for key, value in report_form.items():
                if key != "name" and isinstance(value, str):  # Asosiy xususiyatlarni qo'shish
                    attr_node_id = f"report_attr_{key}"
                    self.knowledge_graph.add_node(attr_node_id, type="report_attribute", name=key, value=value)
                    self.knowledge_graph.add_edge(report_node_id, attr_node_id, relation="has_attribute")
                    self.node_mapping[attr_node_id] = attr_node_id
        
        # Bo'limlar
        if "sections" in self.data:
            for section in self.data["sections"]:
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
        
        logger.info(f"Knowledge Graph yaratildi: {len(self.knowledge_graph.nodes)} tugun, {len(self.knowledge_graph.edges)} bog'lanish")
        return self.knowledge_graph
    
    def save_data_to_json(self, output_file: str) -> None:
        """
        Strukturalangan ma'lumotlarni JSON faylga saqlash.
        
        Args:
            output_file: JSON fayl yo'li
        """
        logger.info(f"Ma'lumotlarni JSON faylga saqlash: {output_file}")
        
        # Agar ma'lumotlar mavjud bo'lmasa, XLSX faylni o'qish
        if not self.data:
            self.read_xlsx()
        
        # JSON faylga saqlash
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Ma'lumotlar muvaffaqiyatli saqlandi: {output_file}")
    
    def visualize_graph(self, output_file: Optional[str] = None) -> None:
        """
        Knowledge Graph ni vizualizatsiya qilish.
        
        Args:
            output_file: Rasm fayl yo'li (ixtiyoriy)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Agar graf mavjud bo'lmasa, uni yaratish
            if not self.knowledge_graph or len(self.knowledge_graph.nodes) == 0:
                self.build_knowledge_graph()
            
            # Graf vizualizatsiyasi
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(self.knowledge_graph)
            
            # Tugun turlariga qarab ranglarni belgilash
            node_colors = []
            for node in self.knowledge_graph.nodes():
                node_type = self.knowledge_graph.nodes[node].get('type', '')
                if node_type == 'report_form':
                    node_colors.append('red')
                elif node_type == 'section':
                    node_colors.append('blue')
                elif node_type == 'column':
                    node_colors.append('green')
                elif node_type == 'row':
                    node_colors.append('orange')
                elif node_type == 'cell':
                    node_colors.append('purple')
                elif node_type in ['strict_control', 'non_strict_control']:
                    node_colors.append('brown')
                else:
                    node_colors.append('gray')
            
            # Grafni chizish
            nx.draw(self.knowledge_graph, pos, with_labels=True, node_color=node_colors, 
                    node_size=500, font_size=8, font_weight='bold', arrows=True)
            
            # Rasmni saqlash yoki ko'rsatish
            if output_file:
                plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
                logger.info(f"Graf vizualizatsiyasi saqlandi: {output_file}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib kutubxonasi o'rnatilmagan. Vizualizatsiya amalga oshirilmadi.")
        except Exception as e:
            logger.error(f"Vizualizatsiya jarayonida xatolik: {str(e)}")
    
    def _extract_important_words(self, so_rov: str) -> List[str]:
        """
        So'rovdan muhim so'zlarni ajratib olish.
        
        Args:
            so_rov: Qidiruv so'rovi
            
        Returns:
            List[str]: Muhim so'zlar ro'yxati
        """
        # So'rovni kichik harflarga o'tkazish va bo'sh joylarni tozalash
        so_rov = so_rov.lower().strip()
        
        # Stop-so'zlar (qidiruv uchun muhim bo'lmagan so'zlar)
        stop_words = {
            'va', 'bilan', 'uchun', 'ham', 'lekin', 'ammo', 'yoki', 'agar', 'chunki',
            'shuning', 'bo\'yicha', 'haqida', 'kerak', 'mumkin', 'qanday', 'qaysi',
            'qanaqa', 'qachon', 'nima', 'kim', 'necha', 'qancha', 'nechta', 'qayerda',
            'qayerga', 'qayerdan', 'qayer', 'qani', 'qay'
        }
        
        # So'zlarni ajratish
        words = so_rov.split()
        
        # Stop-so'zlarni olib tashlash
        important_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return important_words
    
    def search(self, so_rov: str) -> List[Dict]:
        """
        Knowledge Graph orqali qidiruv.
        
        Args:
            so_rov: Qidiruv so'rovi
            
        Returns:
            List[Dict]: Qidiruv natijalari
        """
        logger.info(f"Knowledge Graph qidiruvini boshlash: {so_rov}")
        
        # Agar graf mavjud bo'lmasa, uni yaratish
        if not self.knowledge_graph or len(self.knowledge_graph.nodes) == 0:
            self.build_knowledge_graph()
        
        results = []
        
        # So'rovdagi muhim so'zlarni ajratib olish
        important_words = self._extract_important_words(so_rov)
        
        # Bob va ustun qismlarini aniqlash (regex yordamida)
        import re
        
        bob_part = None
        qator_part = None
        ustun_part = None
        
        # Bob qismini aniqlash
        bob_pattern = re.compile(r'(\d+)[-\s]*bob')
        bob_match = bob_pattern.search(so_rov.lower())
        if bob_match:
            bob_part = bob_match.group(1)
            logger.info(f"Bob qismi aniqlandi: {bob_part}")
        
        # Qator/satr qismini aniqlash
        qator_pattern = re.compile(r'(\d+)[-\s]*(qator|satr)')
        qator_match = qator_pattern.search(so_rov.lower())
        if qator_match:
            qator_part = qator_match.group(1)
            logger.info(f"Qator qismi aniqlandi: {qator_part}")
        
        # Ustun qismini aniqlash
        ustun_pattern = re.compile(r'([a-zA-Zа-яА-ЯёЁ\d]+)[-\s]*ustun')
        ustun_match = ustun_pattern.search(so_rov.lower())
        if ustun_match:
            ustun_part = ustun_match.group(1).upper()
            logger.info(f"Ustun qismi aniqlandi: {ustun_part}")
        
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
            
            # Bob, qator, ustun bo'yicha qo'shimcha tekshirish
            if bob_part and node_data.get('bob') == bob_part:
                match_score += 2  # Bob mos kelsa, yuqoriroq ball
            
            if qator_part and node_data.get('row_code') == qator_part:
                match_score += 2  # Qator mos kelsa, yuqoriroq ball
            
            if ustun_part and node_data.get('column') == ustun_part:
                match_score += 2  # Ustun mos kelsa, yuqoriroq ball
            
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
                elif node_type == 'cell':
                    results.append({
                        'turi': 'katakcha',
                        'qator_kodi': node_data.get('row_code', ''),
                        'ustun': node_data.get('column', ''),
                        'bob': node_data.get('bob', ''),
                        'tavsifi': node_data.get('description', ''),
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
        
        logger.info(f"Knowledge Graph qidiruvida {len(enhanced_results)} natija topildi")
        return enhanced_results
    
    def format_search_results(self, results: List[Dict]) -> Dict:
        """
        Qidiruv natijalarini formatlash.
        
        Args:
            results: Qidiruv natijalari
            
        Returns:
            Dict: Formatlangan natijalar
        """
        formatted_results = []
        
        for result in results:
            formatted_result = {
                'nomi': result.get('nomi', ''),
                'tavsifi': result.get('tavsifi', '')
            }
            
            # Qo'shimcha ma'lumotlarni qo'shish
            if 'bob' in result:
                formatted_result['bob'] = result['bob']
            
            if 'kodi' in result:
                formatted_result['kodi'] = result['kodi']
            
            if 'ustun' in result:
                formatted_result['ustun'] = result['ustun']
            
            if 'qator_kodi' in result:
                formatted_result['qator_kodi'] = result['qator_kodi']
            
            if 'turi' in result:
                formatted_result['turi'] = result['turi']
            
            if 'nazorat_turi' in result:
                formatted_result['nazorat_turi'] = result['nazorat_turi']
            
            formatted_results.append(formatted_result)
        
        return {
            "holat": "muvaffaqiyatli",
            "soni": len(formatted_results),
            "natijalar": formatted_results
        }

def main():
    """Asosiy funksiya."""
    # Fayl yo'llarini aniqlash
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_file = os.path.join(current_dir, "12 invest Hisobot (AI chat uchun).xlsx")
    json_output = os.path.join(current_dir, "invest12.json")
    graph_output = os.path.join(current_dir, "invest12_graph.png")
    
    # Invest12KnowledgeGraph klassini yaratish
    kg = Invest12KnowledgeGraph(xlsx_file)
    
    # XLSX fayldan ma'lumotlarni o'qish
    kg.read_xlsx()
    
    # Ma'lumotlarni JSON faylga saqlash
    kg.save_data_to_json(json_output)
    
    # Knowledge Graph yaratish
    kg.build_knowledge_graph()
    
    # Knowledge Graph ni vizualizatsiya qilish
    kg.visualize_graph(graph_output)
    
    logger.info("Jarayon muvaffaqiyatli yakunlandi.")

if __name__ == "__main__":
    main()