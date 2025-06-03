import os
import sys
import re

# Asosiy loyiha katalogini import path ga qo'shish
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CKP Tool ni import qilish
from tools_llm.ckp.ckp_tool_simple import CkpTool

def main():
    print("CKP Tool yuklanmoqda...")
    ckp_tool = CkpTool()
    print("CKP Tool yuklandi.")
    
    print("\nMST (Mahsulotlarning statistik tasniflagichi) ma'lumotlarini qidirish tizimi.")
    print("Chiqish uchun 'exit' yoki 'quit' deb yozing.")
    
    # Foydalanuvchi bilan suhbat tsikli
    while True:
        user_input = input("\nSavol: ")
        
        # Chiqish shartini tekshirish
        if user_input.lower() in ["exit", "quit", "chiqish"]:
            print("Dastur yakunlandi.")
            break
        
        # So'rovni CKP Tool ga yuborish
        try:
            result = ckp_tool.run(user_input)
            print(result)
        except Exception as e:
            print(f"Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    main()
