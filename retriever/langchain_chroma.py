"""
ChromaDB bilan ishlash uchun wrapper
"""
import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
import shutil
from sentence_transformers import SentenceTransformer

# LLM model embeddings uchun
import torch
import re

# GPU mavjudligini tekshirish va device tanlash
device = torch.device('cpu')  # Faqat CPU ishlatish
print(f"Using device: {device}")

class MultilingualEmbeddings:
    """
    Multilingual embedding sinfi
    """
    def __init__(self, max_length: int = 512, batch_size: int = 8):
        self.max_length = max_length
        self.batch_size = batch_size

        try:
            # Model va tokenizer sozlamalari
            model_name = 'intfloat/multilingual-e5-base'
            
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                model_name,
                local_files_only=False,
                cache_dir="./local_models"
            )
            self.model = XLMRobertaModel.from_pretrained(
                model_name,
                local_files_only=False,
                cache_dir="./local_models"
            ).to(device)
        except Exception as e:
            print(f"Model yoki tokenizer yuklashda xatolik: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Ko'p matnlarni batch tarzda embeddingga o'zgartirish."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        ["passage: " + text for text in batch_texts],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(device)
                    
                    outputs = self.model(**inputs)
                    batch_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            except Exception as e:
                print(f"Embedding yaratishda xatolik: {e}")
                empty_embedding = [0.0] * 768
                embeddings.extend([empty_embedding] * len(batch_texts))
        
        return embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        """Token embeddinglarni o'rtacha pooling qilish."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ChromaManager:
    """ChromaDB wrapper"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")
        self.client = None
        self.collection = None
        self.vectorstore = None
        self.model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
        self._initialize_client()

    def _initialize_client(self, max_retries=3, retry_delay=2):
        """ChromaDB client ni ishga tushirish"""
        for attempt in range(max_retries):
            try:
                # ChromaDB client ni yaratish
                self.client = chromadb.PersistentClient(path=self.persist_directory)

                # Embedding function yaratish
                embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='intfloat/multilingual-e5-base',
                    device='cpu'
                )

                # Kolleksiyani olish yoki yaratish
                try:
                    self.collection = self.client.get_collection(
                        name=self.collection_name,
                        embedding_function=embedding_function
                    )
                    print(f"Mavjud kolleksiya ochildi: {self.collection_name}")
                except ValueError:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=embedding_function
                    )
                    print(f"Yangi kolleksiya yaratildi: {self.collection_name}")

                # Vectorstore attributini o'rnatish
                self.vectorstore = self.collection
                
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Ulanishda xatolik: {e}. Qayta urinish {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    print(f"Maksimal urinishlar soni tugadi. Oxirgi xatolik: {e}")
                    raise

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Hujjatlarni kolleksiyaga qo'shish"""
        try:
            metadatas = metadatas or [{} for _ in texts]
            
            # Har bir hujjat uchun unikal ID yaratish
            ids = [f"doc_{i}_{int(time.time())}" for i in range(len(texts))]
            
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
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            # print(f"Qidirish uchun embedding: {query_embedding}")
            
            # Qidirish
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            return results
        except Exception as e:
            print(f"Qidirishda xatolik: {e}")
            return {"documents": [], "metadatas": [], "distances": []}

    def get_all_documents(self) -> Dict[str, List]:
        """Barcha hujjatlarni olish"""
        try:
            return self.collection.get()
        except Exception as e:
            print(f"Hujjatlarni olishda xatolik: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def delete_all_documents(self):
        """Barcha hujjatlarni o'chirish"""
        try:
            if self.collection:
                self.client.delete_collection(self.collection_name)
            self._initialize_client()
            print("Barcha hujjatlar o'chirildi")
            return True
        except Exception as e:
            print(f"Hujjatlarni o'chirishda xatolik: {e}")
            return False

def remove_duplicate_texts(texts: List[str]) -> List[str]:
    """Duplikat matnlarni o'chirish"""
    if not texts:
        return []
        
    seen = set()
    unique_texts = []
    
    for text in texts:
        if not isinstance(text, str):
            continue
            
        # Matnni tozalash va standartlashtirish
        cleaned_text = ' '.join(text.lower().split())
        if cleaned_text not in seen:
            seen.add(cleaned_text)
            unique_texts.append(text)
            
    return unique_texts

# ChromaManager instansiyasini yaratib, barcha funksiyalarga qulay kirishni ta'minlash
chroma_manager = ChromaManager()

# Funksiyalarni eksport qilish
def count_documents():
    """Bazadagi hujjatlar sonini qaytaradi"""
    result = chroma_manager.get_all_documents()
    return len(result["documents"])

def create_collection():
    """Yangi kolleksiya yaratish"""
    return chroma_manager.delete_all_documents()

def add_documents_from_json(json_file_path: str):
    """JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        if not isinstance(data, list):
            data = [data]

        texts, metadatas = [], []
        seen = set()
        
        for doc in data:
            if "content" in doc:
                # Matnni tozalash va standartlashtirish
                cleaned_text = ' '.join(doc["content"].lower().split())
                if cleaned_text not in seen:
                    seen.add(cleaned_text)
                    texts.append(doc["content"])
                    meta = {k: v for k, v in doc.items() if k != "content"}
                    metadatas.append(meta)

        if texts:
            chroma_manager.add_documents(texts, metadatas)
            return True
        return False
    except Exception as e:
        print(f"JSON fayldan o'qishda xatolik: {e}")
        return False

def remove_all_documents():
    """Barcha hujjatlarni o'chirish"""
    return chroma_manager.delete_all_documents()

def search_documents(query: str, n_results: int = 5) -> Dict[str, List]:
    """Hujjatlar orasidan qidirish"""
    return chroma_manager.search_documents(query, n_results)
