import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
import re
import uuid

# GPU mavjudligini tekshirish va device tanlash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomEmbeddingFunction:
    def __init__(self, model_name='Linq-AI-Research/Linq-Embed-Mistral', device='cuda'):
        print(f"Model yuklanmoqda: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        self.model = self.model.to(self.device)
        self.max_length = 4096

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            # Matnni tozalash
            text = self._preprocess_text(text)
            
            # Tokenizatsiya va embedding
            inputs = self.tokenizer(text, 
                                  return_tensors='pt', 
                                  max_length=self.max_length, 
                                  padding=True, 
                                  truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
            embeddings.append(embedding[0].tolist())
        
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
        return len(self.tokenizer.encode(text))

class ChromaManager:
    """ChromaDB wrapper"""
    def __init__(self, collection_name: str = "documents"):
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
                    metadata={"hnsw:space": "cosine"}
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
    
    def search_documents(self, query: str, n_results: int = 5) -> Dict[str, List]:
        """Hujjatlar orasidan qidirish"""
        try:
            # Query embedding ni olish va numpy array ni list ga o'tkazish
            query_embedding = self.embeddings([query])[0]

            # Qidirish
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Qidirishda xatolik: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def delete_all_documents(self):
        """Barcha hujjatlarni o'chirish"""
        try:
            if self.collection:
                self.client.delete_collection(self.collection_name)
            self.__init__()
            print("Barcha hujjatlar o'chirildi")
            return True
        except Exception as e:
            print(f"Hujjatlarni o'chirishda xatolik: {e}")
            return False

def add_documents_from_json(json_file_path: str):
    """JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish"""
    try:
        # 1. JSON faylni o'qish
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"JSON fayl o'qildi. Elementlar soni: {len(data)}")
        
        # 2. Agar yakka element bo'lsa, listga o'tkazish
        if not isinstance(data, list):
            data = [data]
            print("Yakka element listga o'tkazildi")
        
        # 3. Har bir elementni alohida qo'shish
        added_count = 0
        for i, doc in enumerate(data):
            print(f"\nElement {i+1}/{len(data)} qayta ishlanmoqda...")
            
            if "content" in doc:
                # 4. Matnni tozalash
                text = doc["content"].strip()
                
                # 5. Metadata yaratish
                metadata = {k: v for k, v in doc.items() if k != "content"}
                    
                # 6. Har bir hujjat uchun noyub ID yaratish
                doc_id = str(uuid.uuid4())  # UUID yordamida noyub ID

                print(f"Matn uzunligi: {len(text)}")
                print(f"Metadata: {metadata}")
                print(f"Hujjat ID: {doc_id}")
                
                # 7. ChromaDB ga qo'shish
                try:
                    chroma_manager.add_documents(
                        texts=[text],  # Bitta matn
                        metadatas=[{**metadata, 'id': doc_id}],  # Bitta metadata, ID qo'shildi
                        ids=[doc_id]  # Hujjatni noyub ID bilan qo'shish
                    )
                    print(f"Element {i+1} muvaffaqiyatli qo'shildi")
                    added_count += 1
                except Exception as e:
                    print(f"Element {i+1} ni qo'shishda xatolik: {e}")
            else:
                print(f"Element {i+1} da 'content' maydoni topilmadi")
        
        print(f"\nBarcha elementlar qayta ishlandi. {added_count} ta yangi hujjat qo'shildi.")
        return True
    except Exception as e:
        print(f"JSON faylni qayta ishlashda xatolik: {e}")
        return False

# ChromaManager instansiyasini yaratib
chroma_manager = ChromaManager()

# Funksiyalarni eksport qilish
def count_documents():
    """Bazadagi hujjatlar sonini qaytaradi"""
    return chroma_manager.collection.count()

def remove_all_documents():
    """Barcha hujjatlarni o'chirish"""
    return chroma_manager.delete_all_documents()

def search_documents(query: str, n_results: int = 5):
    """Hujjatlar orasidan qidirish"""
    return chroma_manager.search_documents(query, n_results)
