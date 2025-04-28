from retriever.langchain_chroma import count_documents, add_documents_from_json, create_collection, remove_all_documents, search_documents
# from retriever.langchain_chroma_new import count_documents, add_documents_from_json, create_collection, remove_all_documents, search_documents
# from retriever.langchain_chroma_two import count_documents, add_documents_from_json, create_collection, remove_all_documents, search_documents

# json_file_path = "./tayyor_json/rasmiy_statistika_togrisida_qonun.json"
# json_file_path = "./tayyor_json/kadrlar_malakasini_oshirish.json"
# json_file_path = "./tayyor_json/rasmiy_statistikani_tayyorlash _va_tarqatish.json"
# json_file_path = "./tayyor_json/greet.json"
# json_file_path = "./tayyor_json/gender.json"
# json_file_path = "./tayyor_json/statistika_ishlarini_tashkil_etish_va_yuritish.json"
# json_file_path = "./tayyor_json/qumitasi_faoliyatini_tashkil_etish_pq75.json"


# Yangi collection yaratish
# create_collection()

# Barcha hujjatlarning sonini ko'rish
print(f"Jami hujjatlar soni: {count_documents()}")

# Avval barcha hujjatlarni o'chirish
# remove_all_documents()
create_collection()

# JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
add_documents_from_json(json_file_path)

# Barcha hujjatlarning sonini ko'rish
print(f"Jami hujjatlar soni: {count_documents()}")