"""
ChromaDB vector ma'lumotlar bazasini vizualizatsiya qilish uchun kod.
Bu modul ChromaDB kolleksiyalaridagi vektorlarni 2D va 3D ko'rinishda tasvirlash imkonini beradi.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import pandas as pd

# Yorliqlarni joylashtirish uchun qo'shimcha kutubxona
try:
    from adjustText import adjust_text
except ImportError:
    print("adjustText kutubxonasi topilmadi. 'pip install adjustText' buyrug'i bilan o'rnating")

# Asosiy loyiha papkasiga yo'l
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ChromaDB boshqaruvchisini import qilish
from retriever.langchain_chroma import ChromaManager

class ChromaDBVisualizer:
    """
    ChromaDB vektorlarini vizualizatsiya qilish uchun sinf
    """
    
    def __init__(self, collection_name: str = "documents"):
        """
        ChromaDBVisualizer sinfini ishga tushirish
        
        Args:
            collection_name: ChromaDB kolleksiyasining nomi
        """
        self.chroma_manager = ChromaManager(collection_name=collection_name)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def get_embeddings_and_texts(self) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """
        ChromaDB dan embedding, matnlar va metadatalarni olish
        """
        try:
            # ChromaDB dan ma'lumotlarni olish
            try:
                collection_data = self.chroma_manager.collection.get()
                
                if not collection_data['embeddings'] or len(collection_data['embeddings']) == 0:
                    raise ValueError("Kolleksiyada ma'lumotlar topilmadi yoki bo'sh ma'lumotlar qaytarildi")
                    
                embeddings = np.array(collection_data['embeddings'])
                texts = collection_data['documents']
                metadatas = collection_data.get('metadatas', [{}] * len(texts))
                
            except (KeyError, ValueError, AttributeError) as e:
                print(f"To'g'ridan-to'g'ri olishda xatolik: {e}. Boshqa usul bilan urinib ko'rilmoqda...")
                
                # Barcha hujjatlarni olish
                docs = self.chroma_manager.get_all_documents()
                
                if not docs or not docs['documents']:
                    print("Kolleksiyada ma'lumotlar topilmadi")
                    return np.array([]), [], []
                
                # Matnlarni va metadatalarni olish
                texts = docs['documents']
                metadatas = docs.get('metadatas', [{}] * len(texts))
                
                # Embeddinglarni qayta hisoblash
                embeddings = self.chroma_manager.embeddings(texts)
                embeddings = np.array(embeddings)
            
            print(f"Olingan embeddings: {embeddings.shape}, Matnlar soni: {len(texts)}, Metadatalar soni: {len(metadatas)}")
            return embeddings, texts, metadatas
        
        except Exception as e:
            import traceback
            print(f"Embeddings va matnlarni olishda xatolik: {str(e)}")
            print(traceback.format_exc())
            return np.array([]), [], []

    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'tsne', 
                          n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """
        Embedding vektorlarining o'lchamlarini kamaytirish
        
        Args:
            embeddings: Embedding vektorlari
            method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
            n_components: Natijadagi komponentlar soni (2 yoki 3)
            random_state: Tasodifiy holatni belgilash uchun
            
        Returns:
            Kamaytirilgan o'lchamdagi vektorlar
        """
        if embeddings.size == 0:
            return np.array([])
        
        print(f"{method.upper()} yordamida {embeddings.shape[1]} o'lchamdan {n_components} o'lchamga kamaytirish...")
        
        if method.lower() == 'tsne':
            # Perplexity parametri namunalar sonidan kichik bo'lishi kerak
            # Minimal perplexity 2 bo'lishi kerak
            perplexity_value = min(30, max(2, len(embeddings) // 3))
            
            # Agar namunalar soni juda kam bo'lsa
            if len(embeddings) <= 3:
                print(f"TSNE uchun namunalar soni juda kam ({len(embeddings)}). PCA ishlatiladi.")
                method = 'pca'
                reducer = PCA(n_components=n_components, random_state=random_state)
            else:
                reducer = TSNE(n_components=n_components, random_state=random_state, 
                              perplexity=perplexity_value, 
                              max_iter=1000)  # n_iter o'rniga max_iter ishlatiladi
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
        elif method.lower() == 'mds':
            reducer = MDS(n_components=n_components, random_state=random_state, 
                          dissimilarity='euclidean')
        else:
            raise ValueError(f"Noto'g'ri usul: {method}. 'tsne', 'pca', yoki 'mds' dan birini tanlang.")
        
        reduced_data = reducer.fit_transform(embeddings)
        print(f"Dimensiya kamaytirildi: {reduced_data.shape}")
        
        return reduced_data
    
    def visualize_2d_matplotlib(self, embeddings: np.ndarray, texts: List[str], 
                               method: str = 'tsne', title: str = None, 
                               save_path: str = None, show_labels: bool = True,
                               figsize: Tuple[int, int] = (20, 16),
                               label_length: int = 100):
        """
        Embedding vektorlarini 2D ko'rinishda Matplotlib yordamida vizualizatsiya qilish
        
        Args:
            embeddings: Embedding vektorlari
            texts: Matnlar ro'yxati
            method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
            title: Grafik sarlavhasi
            save_path: Grafikni saqlash yo'li
            show_labels: Matn yorliqlarini ko'rsatish
            figsize: Grafik o'lchami
            label_length: Yorliq uzunligi (maksimal belgilar soni)
        """
        if embeddings.size == 0 or len(texts) == 0:
            print("Vizualizatsiya qilish uchun ma'lumotlar yo'q")
            return
        
        # Dimensiyani kamaytirish
        reduced_data = self.reduce_dimensions(embeddings, method, n_components=2)
        
        # Grafik yaratish
        plt.figure(figsize=figsize)
        
        # Nuqtalarni chizish
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                             c=np.arange(len(reduced_data)) % len(self.colors), 
                             cmap=ListedColormap(self.colors), 
                             alpha=0.7, s=120)
        
        # Matn yorliqlarini qo'shish
        if show_labels:
            # Yorliqlar uchun joylashtirish algoritmini yaxshilash
            from adjustText import adjust_text
            texts_annotations = []
            
            for i, txt in enumerate(texts):
                # Matnni qisqartirish (juda uzun bo'lsa)
                short_txt = txt[:label_length] + '...' if len(txt) > label_length else txt
                text = plt.text(reduced_data[i, 0], reduced_data[i, 1], short_txt, 
                             fontsize=9, alpha=0.8, ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.7, pad=2, boxstyle='round,pad=0.5'))
                texts_annotations.append(text)
            
            # Yorliqlarni bir-biridan ajratish
            try:
                adjust_text(texts_annotations, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
            except ImportError:
                print("adjustText kutubxonasi topilmadi. 'pip install adjustText' buyrug'i bilan o'rnating")
        
        # Grafik sarlavhasi va o'qlar
        if title:
            plt.title(title)
        else:
            plt.title(f'ChromaDB Embeddings - {method.upper()} 2D Vizualizatsiyasi')
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.tight_layout()
        
        # Grafikni saqlash
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik saqlandi: {save_path}")
        
        plt.show()
    
    def visualize_3d_plotly(self, embeddings: np.ndarray, texts: List[str], 
                           method: str = 'tsne', title: str = None, 
                           save_path: str = None, hover_data: Dict[str, List] = None,
                           metadatas: List[Dict] = None, color_by: str = None):
        """
        Embedding vektorlarini 3D ko'rinishda Plotly yordamida vizualizatsiya qilish
        
        Args:
            embeddings: Embedding vektorlari
            texts: Matnlar ro'yxati
            method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
            title: Grafik sarlavhasi
            save_path: Grafikni saqlash yo'li
            hover_data: Sichqoncha ustiga kelganda ko'rsatiladigan qo'shimcha ma'lumotlar
            metadatas: Metadatalar ro'yxati
            color_by: Qaysi metadata maydoni bo'yicha rang berish
        """
        if embeddings.size == 0 or len(texts) == 0:
            print("Vizualizatsiya qilish uchun ma'lumotlar yo'q")
            return
        
        # Dimensiyani kamaytirish
        reduced_data = self.reduce_dimensions(embeddings, method, n_components=3)
        
        # DataFrame yaratish
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'z': reduced_data[:, 2],
            'text': [txt[:100] + '...' if len(txt) > 100 else txt for txt in texts],
        })
        
        # Metadatalarni DataFrame ga qo'shish
        if metadatas and len(metadatas) == len(texts):
            # Barcha mavjud metadata kalitlarini aniqlash
            all_keys = set()
            for meta in metadatas:
                if meta:
                    all_keys.update(meta.keys())
            
            # Metadatalarni DataFrame ga qo'shish
            for key in all_keys:
                df[key] = [meta.get(key, "") if meta else "" for meta in metadatas]
            
            # Agar color_by parametri berilgan bo'lsa, shu metadata bo'yicha rang berish
            if color_by and color_by in all_keys:
                # Unikal qiymatlarni olish
                unique_values = df[color_by].unique()
                # Rang berish uchun kategoriya yaratish
                df['color_category'] = df[color_by].astype('category').cat.codes
            else:
                # Agar color_by berilmagan bo'lsa, standart rang berish
                df['color_category'] = np.arange(len(reduced_data)) % len(self.colors)
        else:
            # Metadatalar yo'q bo'lsa, standart rang berish
            df['color_category'] = np.arange(len(reduced_data)) % len(self.colors)
        
        # Qo'shimcha hover ma'lumotlarini qo'shish
        if hover_data:
            for key, values in hover_data.items():
                if len(values) == len(df):
                    df[key] = values
        
        # 3D grafik yaratish
        if title is None:
            title = f'ChromaDB Embeddings - {method.upper()} 3D Vizualizatsiyasi'
        
        # Rang berish uchun parametrlarni aniqlash
        if color_by and color_by in df.columns:
            # Metadata bo'yicha rang berish
            fig = px.scatter_3d(df, x='x', y='y', z='z', color=color_by,
                               hover_name='text', hover_data=[col for col in df.columns if col not in ['x', 'y', 'z', 'text', color_by]],
                               title=title)
        else:
            # Standart rang berish (kategoriya bo'yicha)
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='color_category',
                               color_discrete_sequence=self.colors,
                               hover_name='text', hover_data=[col for col in df.columns if col not in ['x', 'y', 'z', 'text', 'color_category']],
                               title=title)
        
        # Grafik stilini sozlash
        fig.update_traces(marker=dict(size=5, opacity=0.7),
                         selector=dict(mode='markers'))
        
        # Grafikni saqlash
        if save_path:
            fig.write_html(save_path)
            print(f"Grafik saqlandi: {save_path}")
        
        fig.show()
    
    def visualize_similarity_heatmap(self, embeddings: np.ndarray, texts: List[str], 
                                    title: str = None, save_path: str = None,
                                    max_items: int = 30):
        """
        Embedding vektorlari o'rtasidagi o'xshashlikni heatmap ko'rinishida vizualizatsiya qilish
        
        Args:
            embeddings: Embedding vektorlari
            texts: Matnlar ro'yxati
            title: Grafik sarlavhasi
            save_path: Grafikni saqlash yo'li
            max_items: Ko'rsatiladigan maksimal elementlar soni
        """
        if embeddings.size == 0 or len(texts) == 0:
            print("Vizualizatsiya qilish uchun ma'lumotlar yo'q")
            return
        
        # Agar ma'lumotlar juda ko'p bo'lsa, faqat birinchi max_items ni olish
        if len(embeddings) > max_items:
            print(f"Juda ko'p ma'lumotlar ({len(embeddings)}). Faqat birinchi {max_items} ta element ko'rsatiladi.")
            embeddings = embeddings[:max_items]
            texts = texts[:max_items]
        
        # Normallashtirilgan embeddings
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Kosinus o'xshashligini hisoblash
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Matnlarni qisqartirish
        short_texts = [txt[:30] + '...' if len(txt) > 30 else txt for txt in texts]
        
        # Heatmap yaratish
        plt.figure(figsize=(12, 10))
        heatmap = plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(heatmap, label='Kosinus o\'xshashligi')
        
        # Sarlavha va o'qlar
        if title:
            plt.title(title)
        else:
            plt.title('ChromaDB Embeddings - O\'xshashlik Matritsasi')
        
        plt.xticks(np.arange(len(short_texts)), short_texts, rotation=90)
        plt.yticks(np.arange(len(short_texts)), short_texts)
        plt.tight_layout()
        
        # Grafikni saqlash
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik saqlandi: {save_path}")
        
        plt.show()
    
    def visualize_collection(self, method: str = 'tsne', dimension: int = 2, 
                            title: str = None, save_path: str = None,
                            show_labels: bool = True, label_length: int = 100,
                            color_by: str = None):
        """
        ChromaDB kolleksiyasini vizualizatsiya qilish
        
        Args:
            method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
            dimension: Vizualizatsiya o'lchami (2 yoki 3)
            title: Grafik sarlavhasi
            save_path: Grafikni saqlash yo'li
            show_labels: Matn yorliqlarini ko'rsatish (faqat 2D uchun)
            label_length: Yorliq uzunligi (maksimal belgilar soni)
            color_by: Qaysi metadata maydoni bo'yicha rang berish
        """
        # Embeddings, matnlar va metadatalarni olish
        embeddings, texts, metadatas = self.get_embeddings_and_texts()
        
        if embeddings.size == 0 or len(texts) == 0:
            print("Vizualizatsiya qilish uchun ma'lumotlar yo'q")
            return
        
        # Vizualizatsiya qilish
        if dimension == 2:
            self.visualize_2d_matplotlib(embeddings, texts, method, title, save_path, show_labels, label_length=label_length)
        elif dimension == 3:
            self.visualize_3d_plotly(embeddings, texts, method, title, save_path, metadatas=metadatas, color_by=color_by)
        else:
            raise ValueError(f"Noto'g'ri o'lcham: {dimension}. 2 yoki 3 bo'lishi kerak.")

    def visualize_query_results(self, query: str, limit: int = 5, method: str = 'tsne',
                               dimension: int = 2, title: str = None, save_path: str = None,
                               min_results: int = 3):
        """
        So'rov natijalarini vizualizatsiya qilish
        
        Args:
            query: So'rov matni
            limit: Natijalar soni
            method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
            dimension: Vizualizatsiya o'lchami (2 yoki 3)
            title: Grafik sarlavhasi
            save_path: Grafikni saqlash yo'li
        """
        try:
            # So'rov natijalarini olish
            results = self.chroma_manager.search_documents(query, limit=limit)
            
            if not results:
                print(f"So'rov uchun natijalar topilmadi: {query}")
                return
            
            # Agar natijalar soni juda kam bo'lsa, ko'proq natijalarni olish
            if len(results) < min_results:
                print(f"Natijalar soni juda kam ({len(results)}). Ko'proq natijalar olinmoqda...")
                results = self.chroma_manager.search_documents(query, limit=max(10, limit * 2))
                
            print(f"So'rov: '{query}' uchun {len(results)} ta natija topildi")
            
            # So'rov embeddingi
            query_embedding = self.chroma_manager.embeddings(query)
            
            # Natijalar embeddinglarini olish
            result_embeddings = self.chroma_manager.embeddings(results)
            
            # Barcha embeddinglarni birlashtirish
            all_embeddings = np.vstack([np.array([query_embedding]), np.array(result_embeddings)])
            
            # Matnlarni tayyorlash
            all_texts = [f"SO'ROV: {query}"] + results
            
            # Ranglarni tayyorlash
            colors = ['red'] + ['blue'] * len(results)
            
            # Vizualizatsiya qilish
            if dimension == 2:
                # Dimensiyani kamaytirish
                reduced_data = self.reduce_dimensions(all_embeddings, method, n_components=2)
                
                # Grafik yaratish
                plt.figure(figsize=(12, 10))
                
                # Nuqtalarni chizish
                for i, (x, y) in enumerate(reduced_data):
                    plt.scatter(x, y, c=colors[i], alpha=0.7, s=100 if i == 0 else 70)
                
                # Matn yorliqlarini qo'shish
                # Yorliqlar uchun joylashtirish algoritmini yaxshilash
                from adjustText import adjust_text
                texts_annotations = []
                
                for i, txt in enumerate(all_texts):
                    # Matnni qisqartirish (juda uzun bo'lsa)
                    short_txt = txt[:100] + '...' if len(txt) > 100 else txt
                    text = plt.text(reduced_data[i, 0], reduced_data[i, 1], short_txt, 
                                 fontsize=9, alpha=0.8, ha='center', va='center',
                                 bbox=dict(facecolor='white', alpha=0.7, pad=2, boxstyle='round,pad=0.5'))
                    texts_annotations.append(text)
                
                # Yorliqlarni bir-biridan ajratish
                try:
                    adjust_text(texts_annotations, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
                except ImportError:
                    print("adjustText kutubxonasi topilmadi. 'pip install adjustText' buyrug'i bilan o'rnating")
                
                # Grafik sarlavhasi va o'qlar
                if title:
                    plt.title(title)
                else:
                    plt.title(f"So'rov Natijalarining {method.upper()} 2D Vizualizatsiyasi")
                
                plt.xlabel(f'{method.upper()} Component 1')
                plt.ylabel(f'{method.upper()} Component 2')
                plt.tight_layout()
                
                # Grafikni saqlash
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Grafik saqlandi: {save_path}")
                
                plt.show()
                
            elif dimension == 3:
                # Dimensiyani kamaytirish
                reduced_data = self.reduce_dimensions(all_embeddings, method, n_components=3)
                
                # DataFrame yaratish
                df = pd.DataFrame({
                    'x': reduced_data[:, 0],
                    'y': reduced_data[:, 1],
                    'z': reduced_data[:, 2],
                    'text': [txt[:100] + '...' if len(txt) > 100 else txt for txt in all_texts],
                    'type': ['So\'rov'] + ['Natija'] * len(results)
                })
                
                # 3D grafik yaratish
                if title is None:
                    title = f"So'rov Natijalarining {method.upper()} 3D Vizualizatsiyasi"
                
                fig = px.scatter_3d(df, x='x', y='y', z='z', color='type',
                                   color_discrete_map={'So\'rov': 'red', 'Natija': 'blue'},
                                   hover_name='text',
                                   title=title)
                
                # Grafik stilini sozlash
                fig.update_traces(marker=dict(size=6, opacity=0.8),
                                 selector=dict(mode='markers'))
                
                # Grafikni saqlash
                if save_path:
                    fig.write_html(save_path)
                    print(f"Grafik saqlandi: {save_path}")
                
                fig.show()
            
            else:
                raise ValueError(f"Noto'g'ri o'lcham: {dimension}. 2 yoki 3 bo'lishi kerak.")
        
        except Exception as e:
            print(f"So'rov natijalarini vizualizatsiya qilishda xatolik: {str(e)}")

# Asosiy funksiyalar
def visualize_collection(collection_name: str = "documents", method: str = "tsne", 
                        dimension: int = 2, title: str = None, save_path: str = None,
                        show_labels: bool = True, label_length: int = 100,
                        color_by: str = None):
    """
    ChromaDB kolleksiyasini vizualizatsiya qilish
    
    Args:
        collection_name: ChromaDB kolleksiyasining nomi
        method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
        dimension: Vizualizatsiya o'lchami (2 yoki 3)
        title: Grafik sarlavhasi
        save_path: Grafikni saqlash yo'li
        show_labels: Matn yorliqlarini ko'rsatish (faqat 2D uchun)
        color_by: Qaysi metadata maydoni bo'yicha rang berish
    """
    try:
        visualizer = ChromaDBVisualizer(collection_name=collection_name)
        visualizer.visualize_collection(method, dimension, title, save_path, show_labels, label_length, color_by)
    except Exception as e:
        import traceback
        print(f"Kolleksiyani vizualizatsiya qilishda xatolik: {str(e)}")
        print(traceback.format_exc())

def visualize_query_results(query: str, collection_name: str = "documents", 
                           limit: int = 10, method: str = "tsne", dimension: int = 2,
                           title: str = None, save_path: str = None,
                           min_results: int = 3):
    """
    So'rov natijalarini vizualizatsiya qilish
    
    Args:
        query: So'rov matni
        collection_name: ChromaDB kolleksiyasining nomi
        limit: Natijalar soni
        method: Dimensiyani kamaytirish usuli ('tsne', 'pca', yoki 'mds')
        dimension: Vizualizatsiya o'lchami (2 yoki 3)
        title: Grafik sarlavhasi
        save_path: Grafikni saqlash yo'li
    """
    try:
        visualizer = ChromaDBVisualizer(collection_name=collection_name)
        visualizer.visualize_query_results(query, limit, method, dimension, title, save_path, min_results)
    except Exception as e:
        import traceback
        print(f"So'rov natijalarini vizualizatsiya qilishda xatolik: {str(e)}")
        print(traceback.format_exc())

def visualize_similarity_heatmap(collection_name: str = "documents", title: str = None,
                               save_path: str = None, max_items: int = 30):
    """
    Embedding vektorlari o'rtasidagi o'xshashlikni heatmap ko'rinishida vizualizatsiya qilish
    
    Args:
        collection_name: ChromaDB kolleksiyasining nomi
        title: Grafik sarlavhasi
        save_path: Grafikni saqlash yo'li
        max_items: Ko'rsatiladigan maksimal elementlar soni
    """
    try:
        visualizer = ChromaDBVisualizer(collection_name=collection_name)
        embeddings, texts, _ = visualizer.get_embeddings_and_texts()
        if len(embeddings) > 0 and len(texts) > 0:
            visualizer.visualize_similarity_heatmap(embeddings, texts, title, save_path, max_items)
        else:
            print("Heatmap yaratish uchun ma'lumotlar topilmadi")
    except Exception as e:
        import traceback
        print(f"O'xshashlik matritsasini vizualizatsiya qilishda xatolik: {str(e)}")
        print(traceback.format_exc())

# Misol uchun kod
if __name__ == "__main__":
    # ChromaDB kolleksiyasini vizualizatsiya qilish
    # visualize_collection(collection_name="documents", method="tsne", dimension=2,
    #                     title="ChromaDB Embeddings Visualization", 
    #                     save_path="chroma_visualization_2d.png",
    #                     label_length=100)
    
    # 3D vizualizatsiya (source metadatasi bo'yicha rang berish)
    visualize_collection(collection_name="documents", method="pca", dimension=3,
                        title="ChromaDB Embeddings 3D Visualization", 
                        save_path="chroma_visualization_3d.html",
                        color_by="source")
    
    # So'rov natijalarini vizualizatsiya qilish
    # visualize_query_results(query="AI va sun'iy intellekt", collection_name="documents",
    #                       limit=20, method="pca", dimension=2,  # TSNE o'rniga PCA ishlatish
    #                       title="So'rov Natijalarining Vizualizatsiyasi",
    #                       save_path="query_results_visualization.png",
    #                       min_results=5)
    
    # O'xshashlik heatmap
    # visualize_similarity_heatmap(collection_name="documents", 
    #                            title="ChromaDB Embeddings - O'xshashlik Matritsasi",
    #                            save_path="similarity_heatmap.png")