import threading
import requests
import random
import json


questions = [
    "Statistik ma'lumot nima?",
    "statistika to'g'risidagi qonun 4-modda nima haqida",
    "Noqatʼiy mantiqiy nazorat talabi nima?",
    "statistika qonun nima haqida",
    "1-kb qanday hisobot?"

]


# 🔹 Javoblar yoziladigan faylni bloklash uchun lock
write_lock = threading.Lock()

# 🔹 Har bir thread bajarayotgan funksiya
def send_request(thread_id):
    url = "http://127.0.0.1:8000/chat/"  # kerakli endpoint
    query = {"query": random.choice(questions)}

    try:
        response = requests.post(url, json=query)
        result = response.json()
        answer = result.get("response", "No answer key")
    except Exception as e:
        answer = f"Error: {str(e)}"

    with write_lock:
        with open("answers.txt", "a", encoding="utf-8") as f:
            f.write(f"[Thread-{thread_id}] Query: {query['query']}\n")
            f.write(f"[Thread-{thread_id}] Answer: {answer}\n\n")

# 🔹 Nechta parallel so'rov yuborilsin
NUM_THREADS = 10

threads = []
for i in range(NUM_THREADS):
    t = threading.Thread(target=send_request, args=(i+1))
    threads.append(t)
    t.start()

# 🔹 Barcha thread tugaguncha kutamiz
for t in threads:
    t.join()

print("✅ Barcha so'rovlar yakunlandi. Natijalar: answers.txt faylida.")