from qdrant_client import QdrantClient
from qdrant_client.http import models

# Qdrant klientini yaratish
client = QdrantClient("localhost", port=6333)

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
        # Qidiruv so'rovini amalga oshirish
        search_result = client.search(
            collection_name="documents",
            query_vector=query,
            limit=limit
        )
        
        # Natijalarni qaytarish
        return [hit.payload.get("text", "") for hit in search_result]
    except Exception as e:
        print(f"Qidiruv xatosi: {str(e)}")
        return []
