from retriever.langchain_chroma import count_documents, create_collection, add_documents_from_json, remove_all_documents, search_documents
from retriever.langchain_chroma import chroma_manager
# json_file_path = "./yangi/malumot.json"
# json_file_path = "./yangi/kb_error.json"
# json_file_path = "./yangi/malumotgreet.json"
# json_file_path = "./yangi/estat.json"
# json_file_path = "./yangi/estat_yoriqnoma.json"
# json_file_path = "./yangi/lex_uz.json"
# json_file_path = "./yangi/qarorlar.json"
# json_file_path = "./yangi/reports_data.json"
# json_file_path = "./yangi/kb_error.json"

# json_file_path = "./yozilganlar/rasmiy_statistika_togrisida_qonun.json"
# json_file_path = "./yozilganlar/kadrlar_malakasini_oshirish.json"
# json_file_path = "./yozilganlar/rasmiy_statistikani_tayyorlash _va_tarqatish.json"
# json_file_path = "./yozilganlar/greet.json"
# json_file_path = "./yozilganlar/gender.json"
json_file_path = "./yozilganlar/statistika_ishlarini_tashkil_etish_va_yuritish.json"


# Kolleksiyani o'chirish va yangisini yaratish
# create_collection()

# Avval barcha hujjatlarni o'chirish
# remove_all_documents()

# Barcha hujjatlarning sonini ko'rish
result = chroma_manager.vectorstore.get()
print(f"Jami hujjatlar soni: {len(result['documents'])}")

# JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
add_documents_from_json(json_file_path)

# Barcha hujjatlarning sonini ko'rish
result = chroma_manager.vectorstore.get()
print(f"Jami hujjatlar soni: {len(result['documents'])}")