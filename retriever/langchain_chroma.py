import os
import json
from typing import List, Dict, Any, Optional
import chromadb
import torch
from sentence_transformers import SentenceTransformer
import re
import uuid
import logging

from retriever.encoder import cross_encode_sort

# GPU mavjudligini tekshirish va device tanlash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomEmbeddingFunction:
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        print(f"Model yuklanmoqda: {model_name}")
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        
        # Local model papkasini yaratish
        local_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'local_models')
        os.makedirs(local_models_dir, exist_ok=True)
        
        # Model nomini local papka uchun moslashtirish
        model_folder_name = model_name.replace('/', '_')
        local_model_path = os.path.join(local_models_dir, model_folder_name)
        
        # Xotira cheklovlarini sozlash
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
            
        self.max_length = 4096
        self.batch_size = 256  # Batch size ni belgilash

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Matnlarni tozalash
        texts = [self._preprocess_text(text) for text in input]
        
        # Batch processing
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            # Batch encoding with normalization and truncation
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                max_length=self.max_length,
                truncation=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings.tolist())
        
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
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embeddings = CustomEmbeddingFunction()
        
        # ChromaDB klientini yaratish - persistent_directory bilan
        persistent_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chroma_db')
        os.makedirs(persistent_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persistent_dir)
        
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Kolleksiyani olishda xatolik: {str(e)}")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embeddings,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Kolleksiya yaratildi: {collection_name}")

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

            # Natijalarni birlashtirish
            all_docs = []
            seen_docs = set()

            # Semantic natijalarni qo'shish
            if semantic_results and 'documents' in semantic_results:
                for doc in semantic_results['documents'][0]:
                    if doc not in seen_docs:
                        all_docs.append(doc)
                        seen_docs.add(doc)

            # Keyword natijalarni qo'shish
            if keyword_results and 'documents' in keyword_results:
                for doc in keyword_results['documents'][0]:
                    if doc not in seen_docs:
                        all_docs.append(doc)
                        seen_docs.add(doc)

            # n_results gacha cheklash
            # all_docs = all_docs[:n_results]

            all_docs = cross_encode_sort(query, all_docs, n_results)

            return {"documents": [all_docs], "distances": []}

        except Exception as e:
            print(f"Qidirishda xatolik: {str(e)}")
            return {"documents": [[]], "distances": []}

    def delete_all_documents(self):
        """Barcha hujjatlarni o'chirish"""
        try:
            all_docs = self.collection.get()
            all_ids = all_docs["ids"]
            before = self.collection.count()
            if all_ids:
                self.collection.delete(ids=all_ids)
                after = self.collection.count()
                self.__init__(self.collection_name) 
                print("Hujjatlar o'chirildi.")
                print(f"O'chirishdan oldin: {before}, keyin: {after}")
            else:
                print("O'chirish uchun hujjat yo'q.")
            return True
        except Exception as e:
            print(f"Hujjatlarni o'chirishda xatolik: {e}")
            return False

    def create_collection(self, collection_name: Optional[str] = None):
        """ChromaDB collection yaratish"""
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name if collection_name else self.collection_name,
                embedding_function=self.embeddings,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity uchun
            )
            print(f"Yangi kolleksiya yaratildi: {collection_name if collection_name else self.collection_name}")
        except Exception as e:
            print(f"Kolleksiya yaratishda xatolik: {str(e)}")

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
chroma_manager = ChromaManager("documents")

# Qustionlar uchun alohida ChromaManager yaratish
questions_manager = ChromaManager(collection_name="questions")  # Uncommenting this line

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

def create_collection():
    """ChromaDB collection yaratish"""
    return chroma_manager.create_collection()
