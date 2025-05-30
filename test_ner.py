#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER Test Script - O'zbekiston hududlari uchun Named Entity Recognition testi
"""

import sys
import os

# Loyiha yo'lini qo'shish
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retriever.langchain_chroma import RegionNER, ChromaManager

def test_ner_functionality():
    """NER funksionalligi test qilish"""
    
    print("="*60)
    print("O'ZBEKISTON HUDUDLARI NER TEST DASTURI")
    print("="*60)
    
    # NER instansiyasini yaratish
    ner = RegionNER()
    
    # Test so'rovlari
    test_queries = [
        # MHOBT kodlari bilan
        "MHOBT 1703202 haqida ma'lumot bering",
        "1703 kodi qaysi hudud?",
        "17 raqami nima degani?",
        
        # SOATO kodlari bilan
        "SOATO 1703202001 bo'yicha statistika",
        "12345678901 kodi haqida",
        
        # Hudud nomlari bilan
        "Andijon viloyati haqida",
        "Toshkent shahri aholi soni",
        "Farg'ona viloyatidagi korxonalar",
        "Buxoro viloyati iqtisodiyoti",
        
        # Tuman nomlari bilan
        "Oltinko'l tumani haqida",
        "Asaka tumani statistikasi",
        "Baliqchi tumanidagi aholi",
        
        # Aralash so'rovlar
        "Andijon viloyati 1703 MHOBT kodi bilan",
        "Oltinko'l tumani SOATO 1703202001 statistikasi",
        "MHOBT 1703202 Oltinko'l tumani ma'lumotlari",
        
        # Rus tilida
        "Андижон области информация",
        "Ташкент город население",
        
        # Ingliz tilida
        "Andijan region information",
        "Tashkent city population"
    ]
    
    print("\n📋 TEST SO'ROVLARI VA NER NATIJALARI:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. So'rov: '{query}'")
        
        # NER orqali entitylarni ajratib olish
        entities = ner.extract_entities(query)
        
        print(f"   📍 MHOBT kodlari: {entities['mhobt_codes']}")
        print(f"   📍 SOATO kodlari: {entities['soato_codes']}")
        print(f"   📍 Viloyatlar: {entities['region_names']}")
        print(f"   📍 Tumanlar: {entities['district_names']}")
        
        # Kengaytirilgan so'rov
        enhanced_query = ner.create_enhanced_query(query, entities)
        if enhanced_query != query:
            print(f"   🔍 Kengaytirilgan so'rov: '{enhanced_query[:100]}{'...' if len(enhanced_query) > 100 else ''}'")

def test_search_with_ner():
    """NER bilan qidiruv testi"""
    
    print("\n" + "="*60)
    print("NER BILAN QIDIRUV TESTI")
    print("="*60)
    
    try:
        # ChromaManager instansiyasi
        chroma_manager = ChromaManager("documents")
        
        # Hujjatlar sonini tekshirish
        doc_count = chroma_manager.collection.count()
        print(f"\n📊 Bazada hujjatlar soni: {doc_count}")
        
        if doc_count == 0:
            print("⚠️  Baza bo'sh! Avval JSON fayllardan ma'lumot yuklang.")
            return
        
        # Test so'rovlari
        search_queries = [
            "MHOBT 1703 haqida",
            "Andijon viloyati statistika boshqarmasi",
            "Oltinko'l tumani hudud kodi",
            "1703202 kodi",
        ]
        
        print("\n🔍 QIDIRUV NATIJALARI:")
        print("-" * 60)
        
        for query in search_queries:
            print(f"\n🔎 So'rov: '{query}'")
            
            # NER bilan qidiruv
            results = chroma_manager.search_documents(query, n_results=3)
            
            if results and results.get('documents') and results['documents'][0]:
                print(f"   ✅ Topildi: {len(results['documents'][0])} ta natija")
                
                # NER ma'lumotlarini ko'rsatish
                if 'search_info' in results:
                    search_info = results['search_info']
                    entities = search_info.get('entities_found', {})
                    
                    print(f"   📍 Topilgan entitylar:")
                    if entities.get('mhobt_codes'):
                        print(f"      - MHOBT: {entities['mhobt_codes']}")
                    if entities.get('soato_codes'):
                        print(f"      - SOATO: {entities['soato_codes']}")
                    if entities.get('region_names'):
                        print(f"      - Viloyatlar: {entities['region_names']}")
                    if entities.get('district_names'):
                        print(f"      - Tumanlar: {entities['district_names']}")
                
                # Birinchi natijani ko'rsatish
                first_doc = results['documents'][0][0]
                print(f"   📄 Birinchi natija: {first_doc[:150]}...")
                
                if 'scores' in results and results['scores']:
                    print(f"   🎯 Score: {results['scores'][0]:.3f}")
            else:
                print("   ❌ Hech narsa topilmadi")
                
    except Exception as e:
        print(f"❌ Qidiruv testida xatolik: {str(e)}")

def interactive_test():
    """Interaktiv test rejimi"""
    
    print("\n" + "="*60)
    print("INTERAKTIV NER TEST REJIMI")
    print("="*60)
    print("So'rov kiriting (chiqish uchun 'exit' yozing):")
    
    ner = RegionNER()
    chroma_manager = ChromaManager("documents")
    
    while True:
        try:
            query = input("\n🔍 So'rov: ").strip()
            
            if query.lower() in ['exit', 'quit', 'chiqish']:
                print("👋 Dastur tugatildi!")
                break
                
            if not query:
                continue
            
            print("\n📊 NER TAHLILI:")
            print("-" * 30)
            
            # NER tahlili
            entities = ner.extract_entities(query)
            
            for key, values in entities.items():
                if values and key != 'extracted_text':
                    print(f"{key}: {values}")
            
            # Kengaytirilgan so'rov
            enhanced_query = ner.create_enhanced_query(query, entities)
            if enhanced_query != query:
                print(f"Kengaytirilgan: {enhanced_query}")
            
            # Qidiruv (agar baza mavjud bo'lsa)
            try:
                doc_count = chroma_manager.collection.count()
                if doc_count > 0:
                    print(f"\n🔍 QIDIRUV NATIJALARI:")
                    print("-" * 30)
                    
                    results = chroma_manager.search_documents(query, n_results=2)
                    
                    if results and results.get('documents') and results['documents'][0]:
                        for i, doc in enumerate(results['documents'][0], 1):
                            score = results['scores'][i-1] if 'scores' in results else 0
                            print(f"{i}. Score: {score:.3f}")
                            print(f"   {doc}...")
                    else:
                        print("Hech narsa topilmadi")
                else:
                    print("\n📝 Eslatma: Baza bo'sh, faqat NER tahlili ko'rsatildi")
                    
            except Exception as search_error:
                print(f"Qidiruv xatoligi: {search_error}")
                
        except KeyboardInterrupt:
            print("\n👋 Dastur to'xtatildi!")
            break
        except Exception as e:
            print(f"❌ Xatolik: {str(e)}")

if __name__ == "__main__":
    print("🚀 NER va Qidiruv Test Dasturi")
    
    # Asosiy testlar
    # test_ner_functionality()
    
    # # Qidiruv bilan test
    # test_search_with_ner()
    
    # Interaktiv rejim
    choice = input("\n❓ Interaktiv rejimni ishga tushirish? (y/n): ").strip().lower()
    if choice in ['y', 'yes', 'ha']:
        interactive_test()
    
    print("\n✅ Barcha testlar yakunlandi!")