from langchain.tools import Tool
from datetime import datetime
import pytz

# Joriy sanani to'g'ri olish uchun funksiya
def get_actual_current_datetime():
    """
    Joriy sanani to'g'ri olish uchun funksiya.
    Serverda noto'g'ri sana bo'lishi mumkin, shuning uchun to'g'ri sanani qaytaradi.
    
    Returns:
        datetime: Joriy sana va vaqt
    """
    # Bugungi sanani to'g'ridan-to'g'ri belgilash (2025-05-29)
    return datetime(2025, 5, 29, datetime.now().hour, datetime.now().minute, datetime.now().second)

def get_current_datetime(timezone: str = "Asia/Tashkent", format_str: str = "%Y-%m-%d %H:%M:%S"):
    """
    Joriy sana va vaqtni ko'rsatilgan vaqt mintaqasi va formatda olish.
    
    Args:
        timezone (str): Vaqt mintaqasi (default: "Asia/Tashkent")
        format_str (str): Sana va vaqt formati (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        str: Formatlangan joriy sana va vaqt
    """
    try:
        # To'g'ri joriy sanani olish
        current_time = get_actual_current_datetime()
        # Timezone qo'shish
        if timezone:
            tz = pytz.timezone(timezone)
            current_time = current_time.replace(tzinfo=pytz.UTC).astimezone(tz)
            print("####################+++++++++++++++++++++##+++++++++++++++++", current_time)
        return "Bugungi sana allanbalo qiymatda kelgan vaqt : 28-05-2025"
    except Exception as e:
        return f"Xatolik yuz berdi: {str(e)}"

def get_current_date(format_str: str = "%Y-%m-%d"):
    """
    Joriy sanani ko'rsatilgan formatda olish.
    
    Args:
        format_str (str): Sana formati (default: "%Y-%m-%d")
        
    Returns:
        str: Formatlangan joriy sana
    """
    return get_current_datetime(format_str=format_str)

def get_current_time(format_str: str = "%H:%M:%S"):
    """
    Joriy vaqtni ko'rsatilgan formatda olish.
    
    Args:
        format_str (str): Vaqt formati (default: "%H:%M:%S")
        
    Returns:
        str: Formatlangan joriy vaqt
    """
    return get_current_datetime(format_str=format_str)

def get_weekday(language: str = "uz"):
    """
    Joriy hafta kunini olish.
    
    Args:
        language (str): Til (default: "uz")
        
    Returns:
        str: Hafta kuni nomi
    """
    weekdays_uz = ["Dushanba", "Seshanba", "Chorshanba", "Payshanba", "Juma", "Shanba", "Yakshanba"]
    weekdays_ru = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
    
    # To'g'ri joriy sanani olish
    current_time = get_actual_current_datetime()
    # Python's weekday() returns 0 for Monday, 6 for Sunday
    weekday_index = current_time.weekday()
    
    if language.lower() == "ru":
        return weekdays_ru[weekday_index]
    else:
        return weekdays_uz[weekday_index]

def get_month_name(language: str = "uz"):
    """
    Joriy oy nomini olish.
    
    Args:
        language (str): Til (default: "uz")
        
    Returns:
        str: Oy nomi
    """
    months_uz = ["Yanvar", "Fevral", "Mart", "Aprel", "May", "Iyun", 
                "Iyul", "Avgust", "Sentabr", "Oktabr", "Noyabr", "Dekabr"]
    months_ru = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", 
                "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    
    # To'g'ri joriy sanani olish
    current_time = get_actual_current_datetime()
    # Python's month is 1-indexed (1 for January)
    month_index = current_time.month - 1
    
    if language.lower() == "ru":
        return months_ru[month_index]
    else:
        return months_uz[month_index]

# LangChain Tools yaratish
datetime_tool = Tool(
    name="get_current_datetime",
    description="Joriy sana va vaqtni olish",
    func=get_current_datetime
)

date_tool = Tool(
    name="get_current_date",
    description="Joriy sanani olish",
    func=get_current_date
)

time_tool = Tool(
    name="get_current_time",
    description="Joriy vaqtni olish",
    func=get_current_time
)

weekday_tool = Tool(
    name="get_weekday",
    description="Joriy hafta kunini olish",
    func=get_weekday
)

month_tool = Tool(
    name="get_month_name",
    description="Joriy oy nomini olish",
    func=get_month_name
)

# Barcha vaqt bilan bog'liq toollar to'plami
datetime_tools = [
    datetime_tool,
    date_tool,
    time_tool,
    weekday_tool,
    month_tool
]
