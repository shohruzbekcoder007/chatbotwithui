from retriever.langchain_chroma import questions_manager

import os
import time
import json

def process_all_json_files(folder_path: str = "./tayyor_json/questions/"):
    questions_manager.delete_all_documents()
    questions_manager.create_collection()
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            full_path = os.path.join(folder_path, file)
            
            with open(full_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            print(f"JSON fayl o'qildi. Elementlar soni: {len(data)}")
            
            if not isinstance(data, list):
                data = [data]
            
            data = [item["content"] for item in data]
            print(f"Elementlar soni: {len(data)}")
            print("Yakka element listga o'tkazildi")

            if data:
                questions_manager.add_documents(texts=data, metadatas=[{"type": "question"} for _ in data])
                print(f"{len(data)} ta savol muvaffaqiyatli qo'shildi.")
            else:
                print("Qo'shish uchun savollar mavjud emas.")
        
        print(f"{file} fayli qayta ishlash tugadi.")

process_all_json_files("./tayyor_json/questions/")


query = """Noqat’iy mantiqiy nazorat talabiga ko‘ra, hisobotlar to‘liq va aniq bo‘lishi kerak. Bu shuni anglatadiki, hisobotlarda ko‘rsatilgan ma’lumotlar real holatda mavjud bo‘lishi kerak. Agar hisobotlarda ko‘rsatilgan ma’lumotlar real holatda mavjud bo‘lmasa, bu noqat’iy mantiqiy nazorat talabiga zid bo‘ladi.
Noqat’iy mantiqiy nazorat shuni talab qiladi, hisobotlar to‘liq va aniq bo‘lishi kerak. Bu shuni anglatadiki, hisobotlarda ko‘rsatilgan ma’lumotlar real holatda mavjud bo‘lishi kerak. Agar hisobotlarda ko‘rsatilgan ma’lumotlar real holatda mavjud bo‘lmasa, bu noqat’iy mantiqiy nazorat talabiga zid bo‘ladi."""
results = questions_manager.search_documents(query=query, n_results=10)
print("Questions natijalari:")
print(results)