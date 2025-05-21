import threading
import requests
import random
import json
import os


questions = [
    {
        "content": "Kimlar statistika hisobotini topshiradi"
    },
    {
        "content": "Men yangi tashkilotman, qaysi hisobotlarni topshiraman"
    },
    {
        "content": "Statistika hisobotlari qaysi sohalar bo‘yicha topshiriladi"
    },
    {
        "content": "Statistik hisobotlarning davriyligi qanday"
    },
    {
        "content": "eStat tizimi qanday funksional imkoniyatlarga ega"
    },
    {
        "content": "eStat tizimiga qanday kirish mumkin"
    },
    {
        "content": "Hisobotni (eStat orqali) qanday yuboraman"
    },
    {
        "content": "Hisobotim qabul qilinganini qanday bilsam bo‘ladi?"
    },
    {
        "content": "Xabarnoma va Ma’lumotnoma nima"
    },
    {
        "content": "Qat’iy va noqat’iy mantiqiy nazorat degani nima"
    },
    {
        "content": "Axborot tizimida hisobot uchun izohni qayerga yozaman?"
    },
    {
        "content": "Axborot tizimida qabul qilingan hisobotimni qanday yuklab olaman?"
    },
    {
        "content": "Tizimga qanday kirish mumkin?"
    },
    {
        "content": "Menda ERI kaliti bor. Tizimga kirish tartibi qanday?"
    },
    {
        "content": "OneID orqali eStat tizimiga kirish mumkinmi?"
    },
    {
        "content": "Yangi hisobotni tizimda qanday yaratib, to‘ldirish mumkin?"
    },
    {
        "content": "Hisobot bo‘limini to‘ldirgandan so‘ng ma’lumotlarni qanday saqlash kerak?"
    },
    {
        "content": "Noqat’iy mantiqiy nazorat talabi nimani anglatadi?"
    },
    {
        "content": "Hisobotda noqat’iy nazorat bajarilmasa, nima qilish kerak?"
    },
    {
        "content": "Qat’iy mantiqiy nazorat talabi nima?"
    },
    {
        "content": "Agar hisobotda qat’iy nazorat talabi bajarilmasa, uni jo‘natish mumkinmi?"
    },
    {
        "content": "Tayyor hisobotni tizimga qanday jo‘natish mumkin?"
    },
    {
        "content": "Hisobotni to‘ldirib bo‘ldim. Endi uni qanday jo‘nataman?"
    },
    {
        "content": "Qoralama bo‘limida qanday hisobotlar saqlanadi?"
    },
    {
        "content": "Hisobotni hali yakunlamagan bo‘lsam, uni qayerdan topaman?"
    },
    {
        "content": "Jo‘natilgan hisobotlar bo‘limida nimani ko‘rish mumkin?"
    },
    {
        "content": "Yuborgan hisobotimni qayta ko‘rmoqchiman. Qayerdan topsam bo‘ladi?"
    },
    {
        "content": "Qaytarilgan hisobotlar bo‘limi nimani anglatadi?"
    },
    {
        "content": "Agar hisobotim qaytarilgan bo‘lsa, bu holatni qayerda ko‘raman?"
    },
    {
        "content": "Xabarnomalar bo‘limi nima vazifani bajaradi?"
    },
    {
        "content": "Hisobotlar holati bo‘yicha bildirishnomalarni qayerdan olsam bo‘ladi?"
    },
    {
        "content": "Mas’ul vakillar bo‘limida qanday ma’lumotlar ko‘rsatiladi?"
    },
    {
        "content": "Hisobot topshirish vakolatini boshqa shaxsga qanday berish mumkin?"
    },
    {
        "content": "Gender tenglik bo‘yicha Maslahat-kengash tarkibida kimlar bor va unga kim rahbarlik qiladi?"
    },
    {
        "content": "Bu gender maslahat-kengashi nima ish qiladi o‘zi?"
    },
    {
        "content": "Ishga qabul qilish va ish joyida ayollar va erkaklar teng imkoniyatga egami? Bu borada maslahat-kengash qanday choralar ko‘ryapti?"
    },
    {
        "content": "Gender tenglik buzilganda fuqarolar shikoyatlari bilan bu kengash shug‘ullanadimi? Ayollar va erkaklar uchun teng sharoitlar yaratish va gender tahlili bo‘yicha qanday ishlar qilinadi?"
    },
    {
        "content": "Bu kengash faqat o‘zi ishlaydimi yoki boshqa tashkilotlar bilan ham hamkorlik qiladimi? Masalan, xalqaro tashkilotlar yoki jamoatchilik bilan aloqasi bormi?"
    },
    {
        "content": "Davlat dasturining ustuvor yo‘nalishlari qanday belgilangan?"
    },
    {
        "content": "Salom, sen kimsan?"
    },
    {
        "content": "Assalomu alaykum."
    },
    {
        "content": "1-kb qanday hisobot?"
    },
    {
        "content": "1-kb shakli hisobotining davriyligi qanday"
    },
    {
        "content": "Statistika to'g'risidagi qonun qanday bo'limlardan tashkil topgan?"
    },
    {
        "content": "1-kb (tib) ning davriyligi qanday"
    },
    {
        "content": "Statistika to'g'risidagi qonun qachon qabul qilingan?"
    },
    {
        "content": "Davlat dasturining ustuvor yo‘nalishlari qanday belgilangan?"
    },
    
]


second_questions = [
#     "4-fx shakli hisobotini kimlar qachon topshirishi belgilangan?",
#     "1-kb shaklini kimlar topshiradi?",
#     "1-kb ning qanday turlari bor?",
#     "1-kb shakli hisobotining davriyligi qanday",
#     "1-kb shakli hisobotini kimlar topshiradi?",
#     "4-miz qanday hisobot?",
#     "4-miz shakli hisobotining davriyligi qanday?",
#     "4-miz shakli hisobotini kimlar topshiradi?",
#     "qanday turdagi hisobotlar mavjud?",
#     "men fermerman qaysi hisobotlarni topshiraman?",
#    "Tayyor hisobotni tizimga qanday jo‘natish mumkin?",

#    ################## 2-qism ##################
#    "Hisobotni to‘ldirib bo‘ldim. Endi uni qanday jo‘nataman?",
#    "Qoralama bo‘limida qanday hisobotlar saqlanadi?",
#    "Hisobotni hali yakunlamagan bo‘lsam, uni qayerdan topaman?",
#    "Jo‘natilgan hisobotlar bo‘limida nimani ko‘rish mumkin?",
#    "Yuborgan hisobotimni qayta ko‘rmoqchiman. Qayerdan topsam bo‘ladi?",
#    "Qaytarilgan hisobotlar bo‘limi nimani anglatadi?",
#    "Agar hisobotim qaytarilgan bo‘lsa, bu holatni qayerda ko‘raman?",
#    "Xabarnomalar bo‘limi nima vazifani bajaradi?",
#    "Hisobotlar holati bo‘yicha bildirishnomalarni qayerdan olsam bo‘ladi?",
#    "Mas’ul vakillar bo‘limida qanday ma’lumotlar ko‘rsatiladi?",
#    "Hisobot topshirish vakolatini boshqa shaxsga qanday berish mumkin?",
#    "Gender tenglik bo‘yicha Maslahat-kengash tarkibida kimlar bor va unga kim rahbarlik qiladi?",
#    "Bu gender maslahat-kengashi nima ish qiladi o‘zi?",
#    "milliy statistika ma’lumotlarining xalqaro qiyosiyligini ta’minlash uchun qanday chora-tadbirlar amalga oshirilishi belgilangan?",
#     "Statistika agentligi loyihaning ijro etuvchi organi sifatida qanday vazifalarni bajarishi belgilangan?",

    ###########################  3-qism ###########################
    "qanday mantiqiy nazorat talablari bor?",
    "Qat’iy mantiqiy nazorat talabi nima?",
    "Qat’iy va noqat’iy mantiqiy nazorat degani nima?",
    "Noqat’iy mantiqiy nazorat talabi nimani anglatadi?",
    "Statistika to'g'risidagi qonun qachon qabul qilingan?",
    "Hisobotda noqat’iy nazorat bajarilmasa, nima qilish kerak?",
    "Taqdim etilgan ma’lumotlar to‘liq bo‘lmagan, sifatsiz yoki belgilangan talablarga javob bermagan taqdirda buyurtmachi qanday choralar ko‘radi?",
    "Statistika kuzatuvi natijalarining to‘liqligi va sifati masalasida kuzatuvchi va buyurtmachi o‘rtasida qanday metodologik talab va nazorat mexanizmlari ko‘zda tutilgan?",
    "Estat 4.0 nima?",
    "estat tizimidan kimlar foydalanadi?",
    "Axborot tizimida qabul qilingan hisobotimni qanday yuklab olaman?",
    "Axborot tizimida hisobot uchun izohni qayerga yozaman?",
    "Hisobot bo‘limini to‘ldirgandan so‘ng ma’lumotlarni qanday saqlash kerak?",
    "Hisobotni to‘ldirib bo‘ldim. Endi uni qanday jo‘nataman?",
    "Tayyor hisobotni tizimga qanday jo‘natish mumkin?",
    "Hisobotni (eStat orqali) qanday yuboraman?",
    "Axborot tizimida hisobot uchun izohni qayerga yozaman?",
    "Hisobot bo‘limini to‘ldirgandan so‘ng ma’lumotlarni qanday saqlash kerak?",
    "Yuborgan hisobotimni qayta ko‘rmoqchiman. Qayerdan topsam bo‘ladi?",
    "Hisobotlar holati bo‘yicha bildirishnomalarni qayerdan olsam bo‘ladi?",
    "MIZ (12-korxona shakliga ilova) shakli hisobotini kimlar qachon topshirishi belgilangan?",
    "MIZ (12-korxona shakliga ilova) shakli qanday hisobot?",
    "Kimlar statistika hisobotini topshiradi?",
    "Rasmiy statistika ma’lumotlarining ochiqligi va sifati hamda ularning xalqaro standartlarga muvofiqligini ta’minlash uchun qanday chora-tadbirlar belgilangan?",
    "7-bob",
    "Hisobotim qabul qilinganini qanday bilsam bo‘ladi?",

]




def get_question(question: str = None):
    # return random.choice(questions).get("content")
    return question

import time
# 🔹 Javoblar yoziladigan faylni bloklash uchun lock
write_lock = threading.Lock()

def save_to_json(thread_id, query, answer):
    json_file = "fine_tuning_data.json"
    data = {"prompt": query, "completion": answer}

    # Fayl mavjud bo'lmasa yoki bo'sh bo'lsa, yangi fayl yaratish
    if not os.path.exists(json_file) or os.path.getsize(json_file) == 0:
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([data], f, ensure_ascii=False, indent=4)
    else:
        # Fayl mavjud bo'lsa, mavjud ma'lumotlarni o'qib, yangi ma'lumot qo'shish
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []  # Agar fayl noto'g'ri formatda bo'lsa, bo'sh ro'yxatni boshlash

        existing_data.append(data)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

# 🔹 Har bir thread bajarayotgan funksiya
def send_request(thread_id, question=None):
    url = "http://127.0.0.1:8000/chat/"  # kerakli endpoint
    query = get_question(question)
    # print(f"[Thread-{thread_id}] Query: {query}")

    try:
        response = requests.post(url, json={"query": query})
        result = response.json()
        answer = result.get("response", "No answer key")
    except Exception as e:
        answer = f"Error: {str(e)}"
        print(f"[ Thread-{thread_id} ] Error: {answer}")
        return

    with write_lock:
        print(f"[ Thread-{thread_id} ] Query: {query}")
        # with open("answers.txt", "a", encoding="utf-8") as f:
        #     f.write(f"[Thread-{thread_id}] Query: {query}\n")
        #     f.write(f"[Thread-{thread_id}] Answer: {answer}\n\n")

        # JSON faylga saqlash
        save_to_json(thread_id, query, answer)



for id, question in enumerate(second_questions, start=1):
    send_request(id, question)
    time.sleep(1)  # Har bir so'rov orasida 0.5 soniya kutish


# 🔹 Nechta parallel so'rov yuborilsin
NUM_THREADS = 5


# threads = []
# for i in range(NUM_THREADS):
#     send_request(i+1)
#     time.sleep(1)  # Har bir so'rov orasida 0.5 soniya kutish
#     t = threading.Thread(target=send_request, args=(i+1,))
#     threads.append(t)
#     t.start()

# # 🔹 Barcha thread tugaguncha kutamiz
# for t in threads:
#     time.sleep(1)  # Har bir so'rov orasida 0.5 soniya kutish
#     t.join()

print("✅ Barcha so'rovlar yakunlandi. Natijalar: fine_tuning_data.json faylida.")

