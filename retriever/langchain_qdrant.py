"""
LangChain bilan Qdrant integratsiyasi
"""
import json
import time
from typing import List, Dict, Any, Optional

# LangChain importlari
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Qdrant importlari
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

class QdrantManager:
    """Qdrant va LangChain integratsiyasi"""
    
    def __init__(self):
        self.host = "localhost"
        self.port = 6333
        self.collection_name = "documents"
        self.embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = self._get_client()
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _get_client(self, max_retries=3, retry_delay=2):
        """
        Qdrant klientini yaratish va ulanishni tekshirish
        """
        for attempt in range(max_retries):
            try:
                client = QdrantClient(self.host, port=self.port)
                # Ulanishni tekshirish
                client.get_collections()
                return client
            except UnexpectedResponse as e:
                if attempt < max_retries - 1:
                    print(f"Ulanishda xatolik: {str(e)}. Qayta urinish...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Qdrant serverga ulanib bo'lmadi: {str(e)}")
            except Exception as e:
                raise Exception(f"Qdrant serverga ulanib bo'lmadi: {str(e)}")
    
    def _initialize_vectorstore(self):
        """Vectorstore ni boshlash"""
        try:
            self.vectorstore = Qdrant(
                client=self.client, 
                collection_name=self.collection_name,
                embeddings=self.embeddings,
            )
        except Exception as e:
            print(f"Vectorstore yaratishda xatolik: {str(e)}")
    
    def create_collection(self):
        """
        Documents kolleksiyasini yaratish
        """
        try:
            # Kolleksiyani o'chirish (agar mavjud bo'lsa)
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            
            # Yangi kolleksiya yaratish
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embeddings.client.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            
            # Vectorstore ni qayta ishga tushirish
            self._initialize_vectorstore()
            
            print("Collection muvaffaqiyatli yaratildi")
            return True
        except Exception as e:
            print(f"Collection yaratishda xatolik: {str(e)}")
            return False
    
    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Hujjatlarni Qdrant ga qo'shish

        Args:
            texts (List[str]): Hujjatlar ro'yxati
            metadatas (Optional[List[Dict[str, Any]]]): Har bir hujjat uchun metadata
        """
        try:
            # LangChain Document obyektlarini yaratish
            documents = [
                Document(page_content=text, metadata=meta if meta else {})
                for text, meta in zip(texts, metadatas if metadatas else [{} for _ in texts])
            ]
            
            # Hujjatlarni qo'shish
            self.vectorstore.add_documents(documents)
            print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
        except Exception as e:
            print(f"Hujjatlarni qo'shishda xatolik: {str(e)}")
    
    def add_documents_from_json(self, json_file_path: str):
        """
        JSON fayldan ma'lumotlarni o'qib Qdrant ga qo'shish

        Args:
            json_file_path (str): JSON fayl yo'li
        """
        try:
            # JSON faylni o'qish
            with open(json_file_path, 'r', encoding='utf-8') as file:
                documents = json.load(file)
            
            # documents list bo'lmasa, uni listga o'rash
            if not isinstance(documents, list):
                documents = [documents]
            
            # Matnlarni va metadatalarni ajratib olish
            texts = []
            metadatas = []
            
            for doc in documents:
                texts.append(doc["content"])
                # Qolgan ma'lumotlarni metadata ga qo'shish
                metadata = {k: v for k, v in doc.items() if k != "content"}
                metadatas.append(metadata)
            
            # Hujjatlarni qo'shish
            self.add_documents(texts, metadatas)
            
            print(f"{len(texts)} ta hujjat JSON fayldan muvaffaqiyatli qo'shildi")
        except FileNotFoundError:
            print(f"Xatolik: {json_file_path} fayli topilmadi")
        except json.JSONDecodeError:
            print("Xatolik: JSON fayl formati noto'g'ri")
        except Exception as e:
            print(f"JSON fayldan hujjatlarni qo'shishda xatolik: {str(e)}")
    
    def remove_all_documents(self):
        """
        Barcha hujjatlarni o'chirish
        """
        try:
            # Kolleksiyani o'chirish va qayta yaratish
            self.create_collection()
            print("Barcha hujjatlar o'chirildi")
        except Exception as e:
            print(f"Hujjatlarni o'chirishda xatolik: {str(e)}")
    
    async def search_documents(self, query: str, limit: int = 2):
        """
        Berilgan so'rov bo'yicha eng mos hujjatlarni topish

        Args:
            query (str): Qidiruv so'rovi
            limit (int, optional): Qaytariladigan natijalar soni. Defaults to 2.

        Returns:
            list: Topilgan hujjatlar ro'yxati
        """
        try:
            # LangChain similarity_search metodi
            documents = self.vectorstore.similarity_search(query, k=limit)
            
            # Natijalarni matn koʻrinishida qaytarish
            return [doc.page_content for doc in documents]
        except Exception as e:
            print(f"Qidiruv xatosi: {str(e)}")
            return []


# QdrantManager instansiyasini yaratib, barcha funksiyalarga qulay kirishni ta'minlash
qdrant_manager = QdrantManager()

# Mavjud funktsiyalarni eksport qilish
create_collection = qdrant_manager.create_collection
add_documents = qdrant_manager.add_documents
add_documents_from_json = qdrant_manager.add_documents_from_json
search_documents = qdrant_manager.search_documents
remove_all_documents = qdrant_manager.remove_all_documents
