"""
LangChain bilan ChromaDB integratsiyasi
"""
import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
# from sklearn.metrics.pairwise import cosine_similarity  # scikit-learn o'rnatilmagan

# LangChain importlari
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM model embeddings uchun
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import re

# GPU mavjudligini tekshirish va device tanlash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NumPy bilan Cosine o'xshashlikni hisoblash 
def cosine_similarity_np(A, B=None):
    """
    NumPy yordamida Cosine o'xshashlikni hisoblash
    A: (n_samples_1, n_features) o'lchamdagi array
    B: (n_samples_2, n_features) o'lchamdagi array, None bo'lsa A ishlatiladi
    Returns: (n_samples_1, n_samples_2) o'lchamdagi o'xshashliklar matritsasi
    """
    if B is None:
        B = A
    
    # Array o'lchamlarini tekshirish va kerak bo'lsa qayta shakllantirish
    A = np.asarray(A)
    B = np.asarray(B)
    
    # Agar A 1D array bo'lsa, uni 2D arrayga o'zgartirish
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    # Agar B 1D array bo'lsa, uni 2D arrayga o'zgartirish
    if B.ndim == 1:
        B = B.reshape(1, -1)
        
    # Norm hisoblash
    A_norm = np.linalg.norm(A, axis=1)
    B_norm = np.linalg.norm(B, axis=1)
    
    # Division by zero oldini olish
    A_norm[A_norm == 0] = 1
    B_norm[B_norm == 0] = 1
    
    # Normallash
    A_normalized = A / A_norm[:, np.newaxis]
    B_normalized = B / B_norm[:, np.newaxis]
    
    # Kosinus o'xshashligini hisoblash (dot product of normalized vectors)
    similarities = np.dot(A_normalized, B_normalized.T)
    
    return similarities

class MultilingualEmbeddings:
    """
    LangChain yoki boshqa NLP vazifalari uchun multilingual embedding sinfi.
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
        
        # Batch processing
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            try:
                with torch.no_grad():
                    # E5 modellar uchun tavsiya etilgan tokenizatsiya
                    inputs = self.tokenizer(
                        ["passage: " + text for text in batch_texts],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Modelni chaqirish va o'rtacha pooling
                    outputs = self.model(**inputs)
                    batch_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    
                    # Normalizatsiya qilish
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Natijalarni ro'yxatga qo'shish
                    embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            except Exception as e:
                print(f"Embedding yaratishda xatolik: {e}")
                # Xatolik bo'lganda bo'sh vektorlarni qaytarish
                empty_embedding = [0.0] * 768  # E5-base modellarining o'lchami 768
                embeddings.extend([empty_embedding] * len(batch_texts))
        
        return embeddings
    
    def _mean_pooling(self, model_output, attention_mask):
        """Token embeddinglarni o'rtacha pooling qilish."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_query(self, text: str) -> List[float]:
        """Bitta matnni embeddingga o'zgartirish."""
        return self.embed_documents([text])[0]

class ChromaManager:
    """ChromaDB va LangChain integratsiyasi"""

    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.persist_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma")
        self.embeddings = MultilingualEmbeddings()
        self.vectorstore = None
        self._initialize_vectorstore()

    def _initialize_vectorstore(self, max_retries=3, retry_delay=2):
        """Vectorstore ni boshlash"""
        for attempt in range(max_retries):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                return
            except Exception as e:
                print(f"[{attempt+1}/{max_retries}] Ulanishda xatolik: {e}")
                time.sleep(retry_delay)
        raise Exception("Chroma DB ga ulanib bo'lmadi: maksimal urinishlardan so'ng ham ulanib bo'lmadi")

    def create_collection(self):
        """Yangi kolleksiya yaratish"""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                print("Eski kolleksiya o'chirildi")

            self._initialize_vectorstore()
            print("Yangi kolleksiya yaratildi")
            return True
        except Exception as e:
            print(f"Kolleksiya yaratishda xatolik: {e}")
            return False

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Hujjatlarni kolleksiyaga qo'shish"""
        try:
            metadatas = metadatas or [{} for _ in texts]
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts, metadatas)
            ]
            self.vectorstore.add_documents(documents)
            print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
        except Exception as e:
            print(f"Hujjatlarni qo'shishda xatolik: {e}")

    def add_documents_from_json(self, json_file_path: str):
        """JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if not isinstance(data, list):
                data = [data]

            texts, metadatas = [], []
            for doc in data:
                if "content" in doc:
                    texts.append(doc["content"])
                    metadatas.append({k: v for k, v in doc.items() if k != "content"})

            self.add_documents(texts, metadatas)
            print(f"{len(texts)} ta hujjat JSON fayldan muvaffaqiyatli qo'shildi")
        except FileNotFoundError:
            print(f"Xatolik: {json_file_path} fayli topilmadi")
        except json.JSONDecodeError:
            print("Xatolik: JSON fayl formati noto'g'ri")
        except Exception as e:
            print(f"JSON fayldan hujjatlarni qo'shishda xatolik: {e}")

    def search_documents(self, query: str, limit: int = 10) -> List[str]:
        """So'rov bo'yicha eng mos hujjatlarni topish"""
        try:
            documents = self.vectorstore.similarity_search(query, k=limit)
            results = [doc.page_content for doc in documents]
            return self.rank_by_query_word_presence_regex(results, query)
        except Exception as e:
            print(f"Qidirishda xatolik: {e}")
            return []

    def rank_by_query_word_presence_regex(self, sentences: List[str], query: str) -> List[str]:
        """Querydagi so'zlar soniga qarab gaplarni tartiblash"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        def count_query_words(sentence):
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            return len(query_words & sentence_words)

        return [
            sentence for sentence in sorted(sentences, key=count_query_words, reverse=True)
            if count_query_words(sentence) > 0
        ]

    def remove_all_documents(self):
        """Barcha hujjatlarni o'chirish"""
        try:
            self.vectorstore.delete_collection()
            self._initialize_vectorstore()
            print("Barcha hujjatlar o'chirildi")
        except Exception as e:
            print(f"Hujjatlarni o'chirishda xatolik: {e}")

    def count_documents(self) -> int:
        """Barcha hujjatlarning soni"""
        try:
            return len(self.vectorstore.get())
        except Exception as e:
            print(f"Hujjatlarning sonini olishda xatolik: {e}")
            return 0

    def remove_duplicate_texts(self, texts: List[str], similarity_threshold: float = 0.9) -> List[str]:
        """Takrorlanadigan matnlarni olib tashlash"""
        try:
            # Agar matnlar ro'yxati bo'sh yoki bitta elementdan iborat bo'lsa, o'zini qaytarish
            if not texts or len(texts) < 2:
                return texts
                
            print(f"Takrorlanishlarni tekshirish: {len(texts)} ta matn...")
            
            # Har bir matn uchun embeddinglarni olish
            embeddings = self.embeddings.embed_documents(texts)
            print(f"Embeddinglar olindi. O'lcham: {np.array(embeddings).shape}")
            
            # O'rtacha embeddinglarni solishtirish uchun kosinus o'xshashligini hisoblash
            similarities = cosine_similarity_np(embeddings)
            print(f"O'xshashliklar hisoblab chiqildi. O'lcham: {similarities.shape}")
            
            # Takrorlanadigan matnlarni olib tashlash
            unique_texts = []
            unique_indices = []
            duplicate_count = 0
            
            for i, text in enumerate(texts):
                is_duplicate = False
                for j in unique_indices:
                    if i != j and similarities[i, j] > similarity_threshold:
                        is_duplicate = True
                        duplicate_count += 1
                        break
                
                if not is_duplicate:
                    unique_texts.append(text)
                    unique_indices.append(i)
            
            print(f"{duplicate_count} ta takrorlangan matn olib tashlandi. {len(unique_texts)} ta unikal matn qoldi.")
            return unique_texts
        
        except Exception as e:
            import traceback
            print(f"Takrorlanadigan matnlarni olib tashlashda xatolik: {str(e)}")
            print(traceback.format_exc())  # Batafsil xatolik tafsilotlari
            return texts

# ChromaManager instansiyasini yaratib, barcha funksiyalarga qulay kirishni ta'minlash
chroma_manager = ChromaManager()

# Mavjud funktsiyalarni eksport qilish
create_collection = chroma_manager.create_collection
add_documents = chroma_manager.add_documents
add_documents_from_json = chroma_manager.add_documents_from_json
search_documents = chroma_manager.search_documents
remove_all_documents = chroma_manager.remove_all_documents
count_documents = chroma_manager.count_documents
remove_duplicate_texts = chroma_manager.remove_duplicate_texts
