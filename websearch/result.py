from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def get_web_text(url):
    """ Berilgan URL sahifasidan matnlarni yig'ib olish """
    # Adding verify=False to disable SSL certificate verification
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    text_list = list(soup.stripped_strings)  # Barcha matnlarni olish
    return text_list

def build_faiss_index(text_list, model):
    """ Matnlarni embeddinglarga o‘tkazib, FAISS indeksi yaratish """
    embeddings = model.encode(text_list, convert_to_numpy=True)  # Vektorlar
    index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS L2 normali indeks
    index.add(embeddings)  # Vektorlarni indeksga qo‘shish
    return index, embeddings

def search_semantic(url, query, top_k=3, context_size=5):
    """ Berilgan so‘rov bo‘yicha sahifada semantik qidiruvni amalga oshirish """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Engil NLP modeli
    text_list = get_web_text(url)  # Sahifadagi matnlarni olish

    if not text_list:
        return ["Sahifadan hech qanday matn topilmadi."]

    index, embeddings = build_faiss_index(text_list, model)  # FAISS indeksi yaratish

    query_embedding = model.encode([query], convert_to_numpy=True)  # Qidiruv so‘zini embedding qilish
    _, nearest_ids = index.search(query_embedding, top_k)  # Eng yaqin matnlarni topish

    results = []
    for idx in nearest_ids[0]:  # Har bir topilgan natijani qayta ishlash
        start = max(0, idx - context_size)  # Oldingi 5 ta matn
        end = min(len(text_list), idx + context_size + 1)  # Keyingi 5 ta matn
        full_context = " ... ".join(text_list[start:end])  # Matnlarni bitta stringga birlashtirish
        results.append(full_context)

    return results

# Misol: Veb-sahifada "AI" bo‘yicha semantik qidiruv qilish
url = "https://stat.uz/uz/haqida/markaziy-apparat"  # O'zingizning sahifangizni qo'ying
query = "Samadov"
search_results = search_semantic(url, query)

for i, res in enumerate(search_results, 1):
    print(f"\nNatija {i}:\n{res}\n")
