from retriever.chroma_ import count_documents, create_collection, add_documents_from_json, remove_all_documents, search_documents
# json_file_path = "malumot.json"
json_file_path = "malumotgreet.json"
# json_file_path = "estat.json"
# json_file_path = "estat_yoriqnoma.json"
# json_file_path = "rasmiy_statistika_togrisida_qonun.json"

# Kolleksiyani yaratish
# create_collection()

# Avval barcha hujjatlarni o'chirish
# remove_all_documents()



# JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
add_documents_from_json(json_file_path)

# Barcha hujjatlarning soni
# print(count_documents())