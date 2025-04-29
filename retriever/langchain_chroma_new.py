import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
import torch
from sentence_transformers import SentenceTransformer
import re
import uuid

# GPU mavjudligini tekshirish va device tanlash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomEmbeddingFunction:
    def __init__(self, model_name='Alibaba-NLP/gte-Qwen2-7B-instruct', device='cuda'):
        print(f"Model yuklanmoqda: {model_name}")
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.max_length = 512  # GritLM modeli uchun

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            # Matnni tozalash
            text = self._preprocess_text(text)
            
            # SentenceTransformer orqali embedding olish
            embedding = self.model.encode(text, 
                                       normalize_embeddings=True,
                                       max_length=self.max_length,
                                       truncation=True)
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Matnni tozalash va formatlash
        
        Args:
            text: Kiruvchi matn
            
        Returns:
            str: Tozalangan matn
        """
        if not isinstance(text, str):
            text = str(text)
            
        # HTML teglarni olib tashlash
        text = re.sub(r'<[^>]+>', '', text)
        
        # Ko'p bo'sh joylarni bitta bo'sh joy bilan almashtirish
        text = ' '.join(text.split())
        
        return text
    
    def get_token_count(self, text: str) -> int:
        """
        Matndagi tokenlar sonini hisoblash
        
        Args:
            text: Kiruvchi matn
            
        Returns:
            int: Tokenlar soni
        """
        return len(self.model.tokenize(text))

class ChromaManager:
    """ChromaDB wrapper"""
    def __init__(self, collection_name: str = "documents_new"):
        self.collection_name = collection_name
        self.embeddings = CustomEmbeddingFunction()
        
        # ChromaDB klientini yaratish - persistent_directory bilan
        persistent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chroma_db')
        os.makedirs(persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persistent_dir)
        
        try:
            # Mavjud kolleksiyani olish yoki yangi yaratish
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embeddings
                )
                print(f"Mavjud kolleksiya ochildi: {collection_name}")
            except:
                # Agar kolleksiya mavjud bo'lmasa, yangi yaratish
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embeddings,
                    metadata={"hnsw:space": "cosine"}  # Cosine similarity uchun
                )
                print(f"Yangi kolleksiya yaratildi: {collection_name}")
            
        except Exception as e:
            print(f"Xatolik yuz berdi: {str(e)}")
            raise e

    def get_all_documents(self) -> Dict:
        """
        Barcha hujjatlarni olish
        """
        try:
            # Kolleksiya hajmini olish
            count = self.collection.count()
            if count == 0:
                print("Kolleksiya bo'sh")
                return {"documents": [], "embeddings": [], "metadatas": []}
            
            print(f"Kolleksiyadan {count} ta hujjat olinmoqda...")            
            # Ma'lumotlarni olish
            result = self.collection.get()
            
            # Ma'lumotlarni tekshirish
            if not result or 'documents' not in result:
                raise ValueError("Noto'g'ri format: documents topilmadi")
                
            if not result['documents']:
                raise ValueError("Bo'sh ma'lumotlar qaytarildi")
            
            print(f"Muvaffaqiyatli olindi: {len(result['documents'])} ta hujjat")
            return result
            
        except Exception as e:
            print(f"Hujjatlarni olishda xatolik: {str(e)}")
            return {"documents": [], "embeddings": [], "metadatas": []}
            
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        """
        Kolleksiyaga yangi hujjatlar qo'shish
        
        Args:
            texts: Matnlar ro'yxati
            metadatas: Metadatalar ro'yxati (ixtiyoriy)
            ids: Hujjat ID lari ro'yxati (ixtiyoriy)
        """
        try:
            if not texts:
                print("Qo'shish uchun hujjatlar yo'q")
                return
            
            # ID lar generatsiya qilish
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]
            
            # Metadatani tekshirish
            if metadatas is None:
                metadatas = [{"source": "unknown"} for _ in texts]
            elif len(metadatas) != len(texts):
                raise ValueError(f"Metadatalar soni ({len(metadatas)}) matnlar soniga ({len(texts)}) mos kelmaydi")
            
            # Har bir metadataga noyub ID qo'shish
            for i, metadata in enumerate(metadatas):
                metadata["doc_id"] = ids[i]
            
            # Hujjatlarni qo'shish
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
            
        except Exception as e:
            print(f"Hujjatlarni qo'shishda xatolik: {e}")
    
    def search_documents(self, query: str, n_results: int = 5):
        """Hujjatlar orasidan qidirish"""
        return self.hybrid_search(query, n_results)

    def hybrid_search(self, query: str, n_results: int = 5):
        """
        Hybrid qidiruv - semantic va keyword qidiruvlarni birlashtirib ishlatish
        """
        try:
            # Semantic search
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Ko'proq natija olish
                include=['documents', 'distances', 'metadatas']
            )

            # Keyword search
            keyword_results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,
                where_document={"$contains": query},  # Exact match qidirish
                include=['documents', 'distances', 'metadatas']
            )

            # Natijalarni birlashtirish va dublikatlarni olib tashlash
            combined_results = {
                'documents': [],
                'distances': [],
                'metadatas': []
            }
            
            seen_docs = set()
            
            # Semantic natijalarni qo'shish
            if semantic_results['documents']:
                for i, doc in enumerate(semantic_results['documents'][0]):
                    if doc not in seen_docs:
                        seen_docs.add(doc)
                        combined_results['documents'].append(doc)
                        combined_results['distances'].append(semantic_results['distances'][0][i])
                        combined_results['metadatas'].append(semantic_results['metadatas'][0][i])

            # Keyword natijalarni qo'shish
            if keyword_results['documents']:
                for i, doc in enumerate(keyword_results['documents'][0]):
                    if doc not in seen_docs and len(combined_results['documents']) < n_results:
                        seen_docs.add(doc)
                        combined_results['documents'].append(doc)
                        combined_results['distances'].append(keyword_results['distances'][0][i])
                        combined_results['metadatas'].append(keyword_results['metadatas'][0][i])

            # Natijalarni n_results gacha cheklash
            combined_results['documents'] = combined_results['documents'][:n_results]
            combined_results['distances'] = combined_results['distances'][:n_results]
            combined_results['metadatas'] = combined_results['metadatas'][:n_results]

            return combined_results

        except Exception as e:
            print(f"Qidirishda xatolik: {str(e)}")
            return {
                'documents': [],
                'distances': [],
                'metadatas': []
            }

    def add_documents_from_json(self, json_file_path: str):
        """
        JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
        
        Args:
            json_file_path: JSON fayl yo'li
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if isinstance(data, list):
                texts = []
                metadatas = []
                
                for item in data:
                    if isinstance(item, dict):
                        # content maydonidan matnni olish
                        text = item.get('content', '')
                        # content maydonidan tashqari barcha maydonlarni metadata ga olish
                        metadata = {k: v for k, v in item.items() if k != 'content'}
                        
                        if text:  # Agar matn bo'sh bo'lmasa
                            texts.append(text)
                            metadatas.append(metadata)
                
                if texts:
                    self.add_documents(texts=texts, metadatas=metadatas)
                    print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
                else:
                    print("Qo'shiladigan hujjatlar topilmadi")
            else:
                print("JSON fayl noto'g'ri formatda: ro'yxat kutilgan")
                
        except Exception as e:
            print(f"Xatolik yuz berdi: {str(e)}")
            raise e

    def create_collection(self):
        """Yangi kolleksiya yaratish"""
        try:
            # Avval eski kolleksiyani o'chirish
            try:
                self.client.delete_collection(self.collection_name)
                print(f"Eski kolleksiya o'chirildi: {self.collection_name}")
            except:
                pass  # Agar kolleksiya mavjud bo'lmasa, xatolikni e'tiborsiz qoldiramiz
            
            # Yangi kolleksiya yaratish
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embeddings,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity uchun
            )
            print(f"Yangi kolleksiya yaratildi: {self.collection_name}")
        except Exception as e:
            print(f"Kolleksiya yaratishda xatolik: {str(e)}")
            raise e

    def delete_collection(self):
        """Kolleksiyani o'chirish"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Kolleksiya o'chirildi: {self.collection_name}")
        except Exception as e:
            print(f"Kolleksiyani o'chirishda xatolik: {str(e)}")

# ChromaManager instansiyasini yaratib
chroma_manager = ChromaManager()

def count_documents():
    """Bazadagi hujjatlar sonini qaytaradi"""
    return chroma_manager.collection.count()

def remove_all_documents(where=None):
    """Barcha hujjatlarni o'chirish"""
    try:
        # Barcha hujjatlarni olish
        results = chroma_manager.collection.get(where=where)
        if results and results['ids']:
            # Agar hujjatlar mavjud bo'lsa, ularni o'chirish
            chroma_manager.collection.delete(ids=results['ids'])
            print(f"{len(results['ids'])} ta hujjat o'chirildi")
        else:
            print("O'chiriladigan hujjatlar topilmadi")
    except Exception as e:
        print(f"Hujjatlarni o'chirishda xatolik: {str(e)}")

def search_documents(query: str, n_results: int = 5):
    """Hujjatlar orasidan qidirish"""
    return chroma_manager.search_documents(query, n_results)

def add_documents_from_json(json_file_path: str):
    """JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish"""
    chroma_manager.add_documents_from_json(json_file_path)

def create_collection():
    """ChromaDB collection yaratish"""
    chroma_manager.create_collection()

def delete_collection():
    """ChromaDB collectionni o'chirish"""
    chroma_manager.delete_collection()
