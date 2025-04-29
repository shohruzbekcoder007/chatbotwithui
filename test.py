from retriever.langchain_chroma import count_documents, add_documents_from_json, create_collection, remove_all_documents, search_documents
# from retriever.langchain_chroma_new import count_documents, add_documents_from_json, create_collection, remove_all_documents, search_documents
# from retriever.langchain_chroma_two import count_documents, add_documents_from_json, create_collection, remove_all_documents, search_documents

# json_file_path = "./tayyor_json/rasmiy_statistika_togrisida_qonun.json"
# json_file_path = "./tayyor_json/kadrlar_malakasini_oshirish.json"
# json_file_path = "./tayyor_json/rasmiy_statistikani_tayyorlash _va_tarqatish.json"
# json_file_path = "./tayyor_json/gender.json"
# json_file_path = "./tayyor_json/statistika_ishlarini_tashkil_etish_va_yuritish.json"
# json_file_path = "./tayyor_json/davlat_va_bojxona_tashqi_savdo.json"
# json_file_path = "./tayyor_json/qumitasi_faoliyatini_tashkil_etish_pq75.json"

# json_file_path = "./tayyor_json/statistika_dasturini_tasdiqlash_qaror.json"
# json_file_path = "./tayyor_json/statistika_elektron_shaklda_taqdim_etish_buyruq.json"
# json_file_path = "./tayyor_json/statistika_elektron_shaklda_taqdim_etish_nizom.json"
# json_file_path = "./tayyor_json/estat.json"
# json_file_path = "./tayyor_json/estat_yoriqnoma.json"
json_file_path = "./tayyor_json/reports_data.json"



# tekshirishga_json = "./tayyor_json/tekshirishga.json"


# Yangi collection yaratish
# create_collection()

# Barcha hujjatlarning sonini ko'rish
print(f"Eski hujjatlar soni: {count_documents()}")

# Avval barcha hujjatlarni o'chirish
# remove_all_documents()
# create_collection()

# JSON fayldan ma'lumotlarni o'qib ChromaDB ga qo'shish
# add_documents_from_json(json_file_path)


import os
def process_all_json_files(folder_path: str = "./tayyor_json/"):
    remove_all_documents()
    create_collection()
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                add_documents_from_json(full_path)
                print(f"######################################  Qo'shilgan fayl: {full_path}")

process_all_json_files("./tayyor_json/")
# Barcha hujjatlarning sonini ko'rish
print(f"Jami hujjatlar soni: {count_documents()}")