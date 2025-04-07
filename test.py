from retriever.chroma_ import count_documents, create_collection, add_documents_from_json, remove_all_documents, search_documents
# json_file_path = "./yozilganlar/malumot.json"
# json_file_path = "./yozilganlar/malumotgreet.json"
# json_file_path = "./yozilganlar/estat.json"
# json_file_path = "./yozilganlar/estat_yoriqnoma.json"
# json_file_path = "./yozilganlar/rasmiy_statistika_togrisida_qonun.json"
# json_file_path = "./yozilganlar/lex_uz.json"
# json_file_path = "./yozilganlar/qarorlar.json"
json_file_path = "./yozilganlar/reports_data.json"

print(count_documents())

# Kolleksiyani yaratish
# create_collection()

# Avval barcha hujjatlarni o'chirish
# remove_all_documents()



# JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
add_documents_from_json(json_file_path)

# Barcha hujjatlarning soni
print(count_documents())