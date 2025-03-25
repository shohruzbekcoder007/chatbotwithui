import chromadb
from sentence_transformers import SentenceTransformer
import time
import os
import json
import uuid

def rank_by_query_word_presence(sentences, query):
    """
    Querydagi so‘zlar soniga qarab gaplarni tartiblash.
    """
    query_words = set(query.lower().split())  # Query so‘zlarini kichik harflarga o‘tkazib, ro‘yxatga ajratish
    
    def count_query_words(sentence):
        sentence_words = set(sentence.lower().split())  # Matndagi so‘zlarni ajratish
        print(len(query_words.intersection(sentence_words)))
        return len(query_words.intersection(sentence_words))  # Query so‘zlarini hisoblash
    
    return sorted(sentences, key=count_query_words, reverse=True)

def rank_by_query_word_presence_regex(sentences, query):
    """
    Querydagi so‘zlar soniga qarab gaplarni tartiblash.
    """
    import re
    query_words = set(re.findall(r'\b\w+\b', query.lower()))  # Query so‘zlarini kichik harflarga o‘tkazib, ro‘yxatga ajratish
    
    def count_query_words(sentence):
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))  # Matndagi so‘zlarni ajratish
        print(len(query_words.intersection(sentence_words)))
        return len(query_words.intersection(sentence_words))  # Query so‘zlarini hisoblash
    
    return [sentence for sentence in sorted(sentences, key=count_query_words, reverse=True) if count_query_words(sentence) > 0]
    
    return sorted(sentences, key=count_query_words, reverse=True)


# Modelni yuklash
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_client(max_retries=3, retry_delay=2):
    """
    Chroma klientini yaratish va ulanishni tekshirish
    """
    persist_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma")
    
    for attempt in range(max_retries):
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            return client
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Ulanishda xatolik: {str(e)}. Qayta urinish...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"Chroma DB ga ulanib bo'lmadi: {str(e)}")

# Chroma klientini yaratish
client = get_client()

def create_collection():
    """
    Documents kolleksiyasini yaratish
    """
    try:
        collection = client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection muvaffaqiyatli yaratildi")
        return collection
    except Exception as e:
        print(f"Collection yaratishda xatolik: {str(e)}")
        return None

def add_documents(texts: list[str]):
    """
    Hujjatlarni kolleksiyaga qo'shish

    Args:
        texts (list[str]): Qo'shiladigan matnlar ro'yxati
    """
    try:
        collection = client.get_or_create_collection("documents")
        # Matnlarni vektorlarga o'zgartirish
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        # Hujjatlarni qo'shish
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=[f"doc_{i}" for i in range(len(texts))]
        )
        print(f"{len(texts)} ta hujjat muvaffaqiyatli qo'shildi")
    except Exception as e:
        print(f"Hujjatlarni qo'shishda xatolik: {str(e)}")

def add_documents_from_json(json_file_path: str):
    """
    JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish

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
        
        collection = client.get_or_create_collection("documents")
        # Matnlarni vektorlarga o'zgartirish
        embeddings = model.encode(texts)
        
        # Hujjatlarni qo'shish
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=[f"doc_{uuid.uuid4().hex}-{i}" for i in range(len(texts))]
        )
        print(f"{len(texts)} ta hujjat JSON fayldan muvaffaqiyatli qo'shildi")
    except FileNotFoundError:
        print(f"Xatolik: {json_file_path} fayli topilmadi")
    except json.JSONDecodeError:
        print("Xatolik: JSON fayl formati noto'g'ri")
    except Exception as e:
        print(f"JSON fayldan hujjatlarni qo'shishda xatolik: {str(e)}")

def search_documents(query: str, limit: int = 10):
    """
    Berilgan so'rov bo'yicha eng mos hujjatlarni topish

    Args:
        query (str): Qidiruv so'rovi
        limit (int, optional): Qaytariladigan natijalar soni. Defaults to 2.

    Returns:
        list: Topilgan hujjatlar ro'yxati
    """
    try:
        collection = client.get_collection("documents")

        # So'rovni vektorga o'zgartirish
        query_embedding = model.encode(query, normalize_embeddings=True)

        # Qidiruv
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=limit
        )

        # Query so'zlariga ko'ra gaplarni tartiblash
        filtered_results = rank_by_query_word_presence_regex(results["documents"][0] if results["documents"] else [], query)
        
        # return results["documents"][0] if results["documents"] else []
        return filtered_results
    except Exception as e:
        print(f"Qidirishda xatolik: {str(e)}")
        return []

def remove_all_documents():
    """
    Barcha hujjatlarni o'chirish
    """
    try:
        collection = client.get_collection("documents")
        # Get all document IDs
        results = collection.get()
        if results and results['ids']:
            # Delete all documents by their IDs
            collection.delete(
                where=None,  # No filter condition
                ids=results['ids']  # All document IDs
            )
            print("Barcha hujjatlar o'chirildi")
    except Exception as e:
        print(f"Hujjatlarni o'chirishda xatolik: {str(e)}")

def count_documents():
    """
    Barcha hujjatlarning soni
    """
    try:
        collection = client.get_collection("documents")
        # Barcha documentlar sonini olish
        count = collection.count()
        return count
    except Exception as e:
        print(f"Hujjatlarning sonini olishda xatolik: {str(e)}")
        return 0
