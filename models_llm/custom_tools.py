from langchain.tools import Tool

# 1. Foydalanuvchi autentifikatsiya tool
# 10 ta foydalanuvchi login va parol ma'lumotlari
USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "password1",
    "user2": "password2", 
    "john": "john2025",
    "mary": "mary2025",
    "alex": "alex2025",
    "sarah": "sarah2025",
    "david": "david2025",
    "emma": "emma2025",
    "michael": "michael2025"
}

def authenticate_user(username: str = "", password: str = ""):
    """
    Foydalanuvchi autentifikatsiyasi uchun tool.
    Foydalanuvchi nomi va parolini tekshiradi.
    
    Args:
        username (str): Foydalanuvchi nomi
        password (str): Foydalanuvchi paroli
        
    Returns:
        str: Autentifikatsiya natijasi
    """
    if not username or not password:
        return "Foydalanuvchi nomi va parol kiritilishi shart!"
    
    if username in USER_CREDENTIALS:
        if USER_CREDENTIALS[username] == password:
            return f"Autentifikatsiya muvaffaqiyatli! Xush kelibsiz, {username}."
        else:
            return "Parol noto'g'ri. Iltimos, qayta urinib ko'ring."
    else:
        return "Bunday foydalanuvchi mavjud emas. Iltimos, ro'yxatdan o'ting."

# 2. Ism-familiya tool
# 5 ta ism va familiya ma'lumotlari
NAME_DATABASE = {
    "Alisher": "Navoiy",
    "Bobur": "Mirzo",
    "Abdulla": "Qodiriy",
    "Gafur": "G'ulom",
    "Erkin": "Vohidov"
}

def get_lastname(firstname: str = ""):
    """
    Berilgan ismga mos familiyani qaytaradi.
    
    Args:
        firstname (str): Ism
        
    Returns:
        str: Familiya
    """
    if not firstname:
        return "Iltimos, ism kiriting!"
    
    firstname = firstname.strip().capitalize()
    
    if firstname in NAME_DATABASE:
        return f"{firstname} {NAME_DATABASE[firstname]}"
    else:
        return f"Kechirasiz, {firstname} ismli shaxs ma'lumotlar bazasida topilmadi."

# 3. Mevalar haqida so'ralganda sabzavotlar haqida javob beruvchi tool
FRUITS_TO_VEGETABLES = {
    "olma": "kartoshka",
    "banan": "sabzi",
    "uzum": "piyoz",
    "apelsin": "bodring",
    "nok": "pomidor",
    "anor": "karam",
    "shaftoli": "baqlajon",
    "qulupnay": "qalampir",
    "tarvuz": "rediska",
    "qovun": "sholg'om"
}

def fruit_to_vegetable_info(fruit: str = ""):
    """
    Meva haqida so'ralganda sabzavot haqida ma'lumot beradi.
    
    Args:
        fruit (str): Meva nomi
        
    Returns:
        str: Sabzavot haqida ma'lumot
    """
    if not fruit:
        return "Iltimos, meva nomini kiriting!"
    
    fruit = fruit.strip().lower()
    
    # Meva nomini tekshirish
    if fruit in FRUITS_TO_VEGETABLES:
        vegetable = FRUITS_TO_VEGETABLES[fruit]
        return f"Siz {fruit} haqida so'radingiz, lekin men sizga {vegetable} haqida ma'lumot beraman. {vegetable.capitalize()} - bu foydali sabzavot bo'lib, oshxonada keng qo'llaniladi."
    else:
        return f"Kechirasiz, {fruit} haqida ma'lumot topilmadi. Boshqa meva nomini kiriting."

# LangChain Tools yaratish
auth_tool = Tool(
    name="authenticate_user",
    description="Foydalanuvchi autentifikatsiyasi uchun tool",
    func=authenticate_user
)

name_tool = Tool(
    name="get_lastname",
    description="Berilgan ismga mos familiyani qaytaradi",
    func=get_lastname
)

fruit_vegetable_tool = Tool(
    name="fruit_to_vegetable_info",
    description="Meva haqida so'ralganda sabzavot haqida ma'lumot beradi",
    func=fruit_to_vegetable_info
)

# Barcha custom toollar to'plami
custom_tools = [
    auth_tool,
    name_tool,
    fruit_vegetable_tool
]
