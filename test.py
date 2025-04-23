from retriever.langchain_chroma import count_documents, create_collection, add_documents_from_json, remove_all_documents, search_documents
# from retriever.langchain_chroma import chroma_manager

# json_file_path = "./yozilganlar/rasmiy_statistika_togrisida_qonun.json"
# json_file_path = "./yozilganlar/kadrlar_malakasini_oshirish.json"
# json_file_path = "./yozilganlar/rasmiy_statistikani_tayyorlash _va_tarqatish.json"
# json_file_path = "./yozilganlar/greet.json"
# json_file_path = "./yozilganlar/gender.json"
json_file_path = "./yozilganlar/statistika_ishlarini_tashkil_etish_va_yuritish.json"

# Barcha hujjatlarning sonini ko'rish
print(f"Jami hujjatlar soni: {count_documents()}")

# Avval barcha hujjatlarni o'chirish
# remove_all_documents()

# JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
add_documents_from_json(json_file_path)

# Barcha hujjatlarning sonini ko'rish
print(f"Jami hujjatlar soni: {count_documents()}")