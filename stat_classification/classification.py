import logging
from typing import Dict, List, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StatisticsEconomicsClassifier:
    """Matnni statistika va iqtisodiyot sohalariga tegishliligini aniqlovchi klassifikator."""
    
    def __init__(self):
        """
        Klassifikatorni ishga tushirish.
        """
        logger.info("Statistika va iqtisodiyot klassifikatori yaratildi")
        
        # Klasslar nomlari
        self.id2label = {0: "Boshqa soha", 1: "Statistika va iqtisodiyot"}
        
        # Statistika va iqtisodiyot sohasiga tegishli kalit so'zlar
        self.keywords = [
            "statistika", "iqtisodiyot", "iqtisod", "moliya", "inflatsiya", "yalpi ichki mahsulot", 
            "YIM", "YaIM", "eksport", "import", "budjet", "soliq", "investitsiya", "daromad", 
            "xarajat", "bozor", "narx", "qiymat", "foiz", "bank", "kredit", "valyuta", "kurs",
            "taqsimot", "o'rtacha", "median", "dispersiya", "standart chetlanish", "korrelatsiya",
            "regression", "trend", "prognoz", "dinamika", "ko'rsatkich", "indeks", "raqam",
            "hisobot", "tahlil", "hisob-kitob", "statistik", "iqtisodiy", "moliyaviy"
        ]
        
        # Yuqori ahamiyatga ega kalit so'zlar (ularning mavjudligi matnni to'g'ridan-to'g'ri soha bilan bog'laydi)
        self.high_importance_keywords = [
            "statistika", "iqtisodiyot", "iqtisod", "inflatsiya", "YaIM", "YIM", 
            "budjet", "soliq", "investitsiya", "statistik ma'lumot", "iqtisodiy ko'rsatkich"
        ]
    
    # Fine-tuning, save_model va load_model metodlari olib tashlandi chunki ular kerak emas
    
    def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Matnni klassifikatsiya qilish (faqat kalit so'zlar orqali).
        
        Args:
            text: Klassifikatsiya qilinadigan matn
        
        Returns:
            Natija lug'ati: {'label': yorliq nomi, 'score': ishonchlilik darajasi, 'is_stat_econ': boolean}
        """
        text_lower = text.lower()
        
        # Yuqori ahamiyatli kalit so'zlarni tekshirish
        high_importance_match = any(keyword in text_lower for keyword in self.high_importance_keywords)
        if high_importance_match:
            label_id = 1
            confidence = 0.9  # Yuqori ishonch
        else:
            # Oddiy kalit so'zlarni tekshirish
            # Nechta kalit so'z topilganini hisoblaymiz
            found_keywords = [keyword for keyword in self.keywords if keyword in text_lower]
            keyword_count = len(found_keywords)
            
            if keyword_count >= 2:
                label_id = 1
                # Kalit so'zlar soni qancha ko'p bo'lsa, ishonch shuncha yuqori
                confidence = min(0.5 + (keyword_count * 0.1), 0.85)
            else:
                label_id = 0
                confidence = 0.7 if keyword_count == 0 else 0.6
        
        result = {
            'label': self.id2label[label_id],
            'score': float(confidence),
            'is_stat_econ': label_id == 1,
            'found_keywords': [keyword for keyword in self.keywords if keyword in text_lower]
        }
        
        return result
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Bir nechta matnlarni klassifikatsiya qilish.
        
        Args:
            texts: Klassifikatsiya qilinadigan matnlar ro'yxati
        
        Returns:
            Natijalar ro'yxati
        """
        return [self.predict(text) for text in texts]


def classify_text(text: str) -> Dict[str, Union[str, float]]:
    """
    Matnni statistika va iqtisodiyot sohasiga tegishliligini aniqlash uchun funksiya.
    
    Args:
        text: Klassifikatsiya qilinadigan matn
    
    Returns:
        Natija lug'ati
    """
    # Singleton pattern - bitta model yaratish
    if not hasattr(classify_text, "classifier"):
        try:
            classify_text.classifier = StatisticsEconomicsClassifier()
            logger.info("Klassifikator yaratildi")
        except Exception as e:
            logger.error(f"Klassifikator yaratishda xatolik: {str(e)}")
            return {"error": str(e)}
    
    return classify_text.classifier.predict(text)


# Test qilish uchun kod
if __name__ == "__main__":
    # Test matnlar
    test_texts = [
        "O'zbekistonda 2023-yilda inflatsiya darajasi 12.5% ni tashkil etdi.",
        "Toshkent viloyatida aholi jon boshiga to'g'ri keladigan YaIM ko'rsatkichi o'sdi.",
        "Bugun havo ochiq, harorat +25 daraja.",
        "Statistika qo'mitasi yangi hisobotni e'lon qildi.",
        "Kitob o'qish foydali mashg'ulot hisoblanadi.",
        "Iqtisodiy o'sish sur'atlari barqaror.",
        "Bugun futbol o'yini bo'ladi.",
        "Aholi turmush darajasi",
        "Mahalla ma'lumotlari",
        "soato kodi",
        "pythonda dasturlash"
    ]
    
    # Klassifikator yaratish
    classifier = StatisticsEconomicsClassifier()
    
    # Har bir matn uchun natijalarni chiqarish
    for text in test_texts:
        result = classifier.predict(text)
        print(f"Matn: {text}")
        print(f"Natija: {result['label']} (ishonch: {result['score']:.2f})")
        found_keywords = ', '.join(result['found_keywords']) if result['found_keywords'] else "yo'q"
        print(f"Topilgan kalit so'zlar: {found_keywords}")

        print("---")
