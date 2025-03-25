from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
import time
import json

# Modelni yuklash
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_client(max_retries=3, retry_delay=2):
    """
    Qdrant klientini yaratish va ulanishni tekshirish
    """
    for attempt in range(max_retries):
        try:
            client = QdrantClient("localhost", port=6333)
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

# Qdrant klientini yaratish
client = get_client()

def create_collection():
    """
    Documents kolleksiyasini yaratish
    """
    try:
        # Kolleksiyani o'chirish (agar mavjud bo'lsa)
        client.delete_collection("documents")
        
        # Yangi kolleksiya yaratish
        client.create_collection(
            collection_name="documents",
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        print("Collection muvaffaqiyatli yaratildi")
    except Exception as e:
        print(f"Collection yaratishda xatolik: {str(e)}")

def add_documents(texts: list[str]):
    """
    Hujjatlarni Qdrant ga qo'shish

    Args:
        texts (list[str]): Hujjatlar ro'yxati
    """
    try:
        # Matnlarni vektorga o'girish
        vectors = model.encode(texts).tolist()
        
        # Hujjatlarni qo'shish
        client.upsert(
            collection_name="documents",
            points=models.Batch(
                ids=list(range(len(texts))),
                vectors=vectors,
                payloads=[{"text": text} for text in texts]
            )
        )
        print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
    except Exception as e:
        print(f"Hujjatlarni qo'shishda xatolik: {str(e)}")

def add_documents_from_json(json_file_path: str):
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
        
        # Matnlarni ajratib olish
        texts = [doc["content"] for doc in documents]
        
        # Hujjatlarni Qdrant ga qo'shish
        add_documents(texts)
        print(f"{len(texts)} ta hujjat JSON fayldan muvaffaqiyatli qo'shildi")
    except FileNotFoundError:
        print(f"Xatolik: {json_file_path} fayli topilmadi")
    except json.JSONDecodeError:
        print("Xatolik: JSON fayl formati noto'g'ri")
    except Exception as e:
        print(f"JSON fayldan hujjatlarni qo'shishda xatolik: {str(e)}")


def remove_all_documents():
    """
    Barcha hujjatlarni o'chirish
    """
    try:
        # Kolleksiyani o'chirish (agar mavjud bo'lsa)
        client.delete_collection("documents")
        print("Barcha hujjatlar o'chirildi")
    except Exception as e:
        print(f"Hujjatlarni o'chirishda xatolik: {str(e)}")

async def search_documents(query: str, limit: int = 2):
    """
    Berilgan so'rov bo'yicha eng mos hujjatlarni topish

    Args:
        query (str): Qidiruv so'rovi
        limit (int, optional): Qaytariladigan natijalar soni. Defaults to 2.

    Returns:
        list: Topilgan hujjatlar ro'yxati
    """
    try:
        # Matnni vektorga o'girish
        query_vector = model.encode(query).tolist()

        # Qidiruv so'rovini amalga oshirish
        search_result = client.search(
            collection_name="documents",
            query_vector=query_vector,
            limit=limit
        )
        
        # Natijalarni qaytarish
        return [hit.payload.get("text", "") for hit in search_result]
    except Exception as e:
        print(f"Qidiruv xatosi: {str(e)}")
        return []
