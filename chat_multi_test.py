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
        "content": "Aholiga davlat xizmatlarini ko‘rsatishda qanday qulayliklar va imtiyozlar yaratiladi?"
    },
    {
        "content": "Ijtimoiy himoya sohasida (pensiya ta’minoti va nogironlik) qanday asosiy vazifalar belgilangan?"
    },
    {
        "content": "Aholini uy-joy bilan ta’minlash borasida qanday maqsad va chora-tadbirlar belgilangan?"
    },
    {
        "content": "Tashabbusli byudjet va mahallalarning moliyaviy mustaqilligini kengaytirish bo‘yicha qanday choralar ko‘zda tutilgan?"
    },
    {
        "content": "Ta’lim sifatini oshirish va sohani isloh qilish borasida dasturda qanday vazifalar belgilangan?"
    },
    {
        "content": "Aholini sifatli tibbiy xizmatlar bilan ta’minlash va kasalliklarning oldini olish borasida qanday tadbirlar belgilangan?"
    },
    {
        "content": "Sog‘lom turmush tarzini shakllantirish bo‘yicha dasturda qanday tashabbuslar nazarda tutilgan?"
    },
    {
        "content": "Tadbirkorlik subyektlarini qo‘llab-quvvatlash va ularning huquqiy himoyasini kuchaytirish bo‘yicha qanday choralar ko‘zda tutilgan?"
    },
    {
        "content": "Qishloq xo‘jaligini suv resurslari bilan barqaror ta’minlash uchun qanday choralar belgilangan?"
    },
    {
        "content": "Sud-huquq tizimini isloh qilish va qonun ustuvorligini ta’minlash bo‘yicha qanday o‘zgarishlar nazarda tutilgan?"
    },
    {
        "content": "Yo‘l harakati xavfsizligini oshirish maqsadida qanday yangi qoidalar joriy etiladi?"
    },
    {
        "content": "Davlat dasturi ijrosini monitoring qilish va tahlil etish bo‘yicha qanday mexanizmlar nazarda tutilgan?"
    },
    {
        "content": "Statistika organlari xodimlari malaka oshirish kurslarida qachon va qay tartibda o‘qitilishi belgilangan?"
    },
    {
        "content": "Respublika va mahalliy ijro hokimiyati organlarining statistika uchun mas’ul xodimlari malaka oshirish kurslarida qanday tartibda ishtirok etadi?"
    },
    {
        "content": "Statistika hisobotlarini topshiruvchi tadbirkorlik subyektlari xodimlarining statistik savodxonligini oshirish uchun malaka oshirish kurslari qanday shaklda tashkil etiladi?"
    },
    {
        "content": "Statistika sohasida kadrlarning uzluksiz kasbiy rivojlanishini ta’minlash uchun qaysi ekspertlar kengashi tuzilishi nazarda tutilgan?"
    },
    {
        "content": "Statistika agentligi tizimidagi barcha xodimlarni qayta tayyorlash qanday tartibda amalga oshirilishi belgilangan?"
    },
    {
        "content": "Kadrlarni qayta tayyorlash va malaka oshirish jarayonlarida masofaviy va boshqa zamonaviy ta’lim shakllarini joriy etish qanday nazarda tutilgan?"
    },
    {
        "content": "Malaka oshirish jarayonlarini raqamli texnologiyalar asosida tashkil etish va ularni monitoring qilish orqali ta’lim sifati qanday oshiriladi?"
    },
    {
        "content": "Statistika xodimlarining bilim va malakasini mustaqil baholash uchun qanday mexanizm joriy qilinadi?"
    },
    {
        "content": "Xodimlarning kasbiy malakasini oshirish natijalarini baholash va ularni rag‘batlantirish tizimi qanday joriy etilishi belgilangan?"
    },
    {
        "content": "Istiqbolli yosh statistika mutaxassislarini qo‘llab-quvvatlash maqsadida kadrlar zaxirasini shakllantirish va ularni stajirovkaga yuborish qanday amalga oshiriladi?"
    },
    {
        "content": "Statistika tizimi xodimlari uchun vebinar tarzidagi amaliy trening va mahorat darslari qanday davriylikda o‘tkazib turilishi rejalashtirilgan?"
    },
    {
        "content": "Mahalliy va xorijiy ekspertlar ishtirokida milliy va xalqaro statistika uslubiyoti bo‘yicha seminarlar hamda davra suhbatlari qanday davriylikda tashkil etib boriladi?"
    },
    {
        "content": "Statistika sohasi bo‘yicha elektron o‘quv platformasi qachondan ishga tushiriladi va u xodimlar malakasini oshirish uchun qanday imkoniyatlarni taqdim etadi?"
    },
    {
        "content": "Malaka oshirish kurslari dasturlarini xalqaro standartlar hamda “Davlat fuqarolik xizmati to‘g‘risida”gi Qonun talablariga muvofiqlashtirish bo‘yicha qanday vazifalar belgilangan?"
    },
    {
        "content": "Kadrlar malakasini oshirish institutining yangi tuzilmasini tasdiqlash va shtat birliklari sonini belgilash qanday tartibda amalga oshiriladi?"
    },
    {
        "content": "Kadrlar malakasini oshirish instituti direktori va uning o‘rinbosarlarini tayinlash hamda institut professor-o‘qituvchilariga mehnatga haq to‘lash masalalari qanday hal etiladi?"
    },
    {
        "content": "Institutdagi xodimlarni attestatsiyadan o‘tkazish uchun qanday tartibda komissiya tuzilishi belgilangan?"
    },
    {
        "content": "Malaka oshirish va qayta tayyorlash kurslarini tamomlagan tinglovchilarga beriladigan hujjatlar qanday tartibda tayyorlanadi va elektron tarzda rasmiylashtiriladi?"
    },
    {
        "content": "Kadrlar malakasini oshirish institutining majlislar zali va yotoqxona binolarini rekonstruksiya qilish hamda jihozlash bo‘yicha qanday ishlar rejalashtirilgan va bu jarayonda qaysi tashkilotlar mas’ul etib belgilangan?"
    },
    {
        "content": "Mazkur kadrlar malakasini oshirish bo'yicha qaror ijrosini nazorat qilish vazifasi kimga yuklatilgan?"
    },
    {
        "content": "Milliy statistika qo‘mitasining davlat statistika sohasidagi asosiy funksiyalari nimalardan iborat va qo‘mitaga qanday vakolatlar berilgan?"
    },
    {
        "content": "statistika qo‘mitasining tashkiliy tuzilmasi va undagi lavozimlar tizimi qanday shakllantirilgan hamda milliy statistika tizimini transformatsiya qilish bosqichlari qanday amalga oshiriladi?"
    },
    {
        "content": "Milliy statistika qo‘mitasi tizimida asosiy samaradorlik ko‘rsatkichlari (KPI) tizimi qanday joriy etilmoqda va xodimlar samaradorligini baholash mezonlari nimalardan iborat?"
    },
    {
        "content": "Rasmiy statistika ma’lumotlarining ochiqligi va sifati hamda ularning xalqaro standartlarga muvofiqligini ta’minlash uchun qanday chora-tadbirlar belgilangan?"
    },
    {
        "content": "Statistika faoliyatini raqamlashtirish hamda sohada zamonaviy axborot tizimlari va monitoring mexanizmlarini joriy etish jarayonlari qanday tashkil etilgan?"
    },
    {
        "content": "Milliy statistika qo‘mitasida kadrlar siyosati qanday amalga oshiriladi, ichki nazorat tizimi qanday yo‘lga qo‘yilgan va korrupsiyaga qarshi kurashish bo‘yicha qanday choralar ko‘riladi?"
    },
    {
        "content": "Milliy statistika qo‘mitasining xalqaro hamkorlik faoliyati va donor tashkilotlar bilan o‘zaro aloqalari qanday yo‘lga qo‘yilgan?"
    },
    {
        "content": "Rasmiy statistika qonuni bo‘yicha Qonunning 4-moddasida keltirilgan “identifikator” tushunchasi nimani anglatadi va statistik birlikni identifikatsiyalashda qanday ro‘l o‘ynaydi?"
    },
    {
        "content": "Rasmiy statistika qonuni bo‘yicha Qonunning 4-moddasiga binoan “individual ma’lumotlar” deb nimani tushunish kerak va u rasmiy statistika maqsadlari uchun qanday ishlatiladi?"
    },
    {
        "content": "Rasmiy statistika qonuni bo‘yicha “Ma’muriy ma’lumotlar” termini qonunning 4-moddasiga muvofiq qanday ma’lumotlarni ifodalaydi va bu ma’lumotlar rasmiy statistika tayyorlash jarayonida qanday foydalaniladi?"
    }
]

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
def send_request(thread_id):
    url = "http://127.0.0.1:8000/chat/"  # kerakli endpoint
    query = random.choice(questions).get("content")
    # print(f"[Thread-{thread_id}] Query: {query}")

    try:
        response = requests.post(url, json={"query": query})
        result = response.json()
        answer = result.get("response", "No answer key")
    except Exception as e:
        answer = f"Error: {str(e)}"
        print(f"[Thread-{thread_id}] Error: {answer}")
        return

    with write_lock:
        print(f"[Thread-{thread_id}] Query: {query}")
        # with open("answers.txt", "a", encoding="utf-8") as f:
        #     f.write(f"[Thread-{thread_id}] Query: {query}\n")
        #     f.write(f"[Thread-{thread_id}] Answer: {answer}\n\n")

        # JSON faylga saqlash
        save_to_json(thread_id, query, answer)

# 🔹 Nechta parallel so'rov yuborilsin
NUM_THREADS = 5

threads = []
for i in range(NUM_THREADS):
    # send_request(i+1)
    # time.sleep(1)  # Har bir so'rov orasida 0.5 soniya kutish
    t = threading.Thread(target=send_request, args=(i+1,))
    threads.append(t)
    t.start()

# 🔹 Barcha thread tugaguncha kutamiz
for t in threads:
    time.sleep(1)  # Har bir so'rov orasida 0.5 soniya kutish
    t.join()

print("✅ Barcha so'rovlar yakunlandi. Natijalar: answers.txt faylida.")