import os
import sys
import re
import torch
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from tools_llm.soato.soato_tool import SoatoTool

# Add the tools_llm directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the SoatoTool
from tools_llm.soato.soato_tool import SoatoTool

def main():
    # Initialize the Ollama LLM
    llm = OllamaLLM(
        base_url="http://localhost:11434",  # Default Ollama URL
        model="devstral",  # Using an available model from Ollama
        temperature=0.7
    )
    
    # Initialize the SOATO tool
    soato_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tools_llm", "soato", "soato.json"
    )
    
    # Check if CUDA is available for GPU acceleration
    use_gpu = torch.cuda.is_available()
    print(f"GPU mavjud: {use_gpu}")
    
    # Create the SOATO tool with embedding support
    print("SOATO Tool yuklanmoqda...")
    soato_tool = SoatoTool(soato_file_path, use_embeddings=True)
    print("SOATO Tool yuklandi.")
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the agent with the SOATO tool
    agent = initialize_agent(
        tools=[soato_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,  # Maksimal takrorlashlar sonini cheklash
        early_stopping_method="generate"  # Parsing xatoligida to'g'ridan-to'g'ri javob generatsiya qilish
    )
    
    print("\nSOATO Tool Test - O'zbekiston ma'muriy hududiy birliklari haqida so'rang")
    print("Embedding model yordamida semantik qidiruv ham ishlaydi")
    print("Chiqish uchun 'exit' yoki 'quit' kiriting")
    print("-" * 50)
    
    # Start the conversation loop
    while True:
        user_input = input("\nSavol: ")
        
        if user_input.lower() in ["exit", "quit", "chiqish"]:
            print("Dastur yakunlandi.")
            break
        
        try:
            response = agent.run(user_input)
            print(f"\nJavob: {response}")
        except Exception as e:
            print(f"\nXatolik yuz berdi: {str(e)}")
            # Agar parsing xatoligi bo'lsa, to'g'ridan-to'g'ri LLM dan javob olish
            try:
                print("\nTo'g'ridan-to'g'ri LLM dan javob olinmoqda...")
                # Kontekst va so'rovni birlashtirish uchun prompt
                if "Could not parse LLM output:" in str(e):
                    # Xatolik xabaridan ma'lumotlarni olish
                    error_text = str(e)
                    # SOATO qidiruv natijalarini olish
                    try:
                        # Xatolik xabaridan SOATO ma'lumotlarini ajratib olish
                        search_results = ""
                        parts = error_text.split("Could not parse LLM output: `")
                        if len(parts) > 1:
                            search_results = parts[1].split("`")[0]
                        
                        # SOATO ma'lumotlarini tozalash
                        search_results = search_results.replace("Thought:", "")
                        search_results = search_results.replace("Observation:", "")
                        search_results = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}.*?"HTTP/1\.1 200 OK"', '', search_results)
                        search_results = re.sub(r'For troubleshooting.*?OUTPUT_PARSING_FAILURE', '', search_results)
                        
                        # SOATO ma'lumotlarini ajratib olish
                        # Avval barcha SOATO ma'lumotlarini o'z ichiga olgan qismni ajratib olish
                        if "SOATO ma'lumotlari:" in search_results:
                            soato_data = re.search(r'SOATO ma\'lumotlari:[\s\S]*?(?=Bu ma\'lumotlar|$)', search_results)
                            if soato_data:
                                search_results = soato_data.group(0).strip()
                        
                        # Ma'lumot turlarini tekshirish va ajratib olish
                        if "TUMAN MA'LUMOTLARI:" in search_results:
                            # Tuman ma'lumotlari formatini ajratib olish
                            tuman_info = re.search(r'TUMAN MA\'LUMOTLARI:[\s\S]*?(?=Boshqa mos natijalar:|$)', search_results)
                            if tuman_info:
                                clean_info = tuman_info.group(0).strip()
                                search_results = clean_info
                        elif "VILOYAT MA'LUMOTLARI:" in search_results:
                            # Viloyat ma'lumotlari formatini ajratib olish
                            viloyat_info = re.search(r'VILOYAT MA\'LUMOTLARI:[\s\S]*?(?=Boshqa mos natijalar:|$)', search_results)
                            if viloyat_info:
                                clean_info = viloyat_info.group(0).strip()
                                search_results = clean_info
                        elif "SHAHAR MA'LUMOTLARI:" in search_results:
                            # Shahar ma'lumotlari formatini ajratib olish
                            shahar_info = re.search(r'SHAHAR MA\'LUMOTLARI:[\s\S]*?(?=Boshqa mos natijalar:|$)', search_results)
                            if shahar_info:
                                clean_info = shahar_info.group(0).strip()
                                search_results = clean_info
                        elif "SOATO kodi:" in search_results:
                            # SOATO kodi bo'lgan qismni ajratib olish
                            soato_info = re.search(r'SOATO kodi:[\s\S]*?(?=Boshqa mos natijalar:|$)', search_results)
                            if soato_info:
                                clean_info = soato_info.group(0).strip()
                                search_results = clean_info
                        
                        # Agar natija topilmagan bo'lsa, o'zbekcha xabar qaytarish
                        if not search_results or len(search_results.strip()) < 10:
                            search_results = f"'{user_input}' so'rovi bo'yicha ma'lumot topilmadi."
                        
                        # Debug uchun ma'lumotlarni ko'rsatish
                        print(f"\nAjratilgan SOATO ma'lumotlari:\n{search_results}\n")
                        
                        # Yaxshilangan prompt
                        prompt = f"""Quyidagi SOATO ma'lumotlari asosida foydalanuvchining '{user_input}' savoliga o'zbek tilida aniq va qisqa javob ber.
                        
                        MUHIM QOIDALAR:
                        1. Javobni faqat o'zbek tilida ber.
                        2. Javobingda faqat eng mos keladigan ma'lumotlarni ishlatib, ortiqcha ma'lumotlarni ko'rsatma.
                        3. Javobni quyidagi formatda tayyorla: "[Hududning to'liq nomi] haqida ma'lumot: [asosiy ma'lumotlar]."
                        4. Agar ma'lumotlar ichida SOATO kodi, aholisi, maydoni kabi ma'lumotlar bo'lsa, ularni ham qo'sh.
                        5. Javobni formatlashda markdown ishlatma, oddiy matn formatida javob ber.
                        
                        Ma'lumotlar:
                        {search_results}
                        """
                    except Exception as extract_error:
                        print(f"Ma'lumotlarni ajratib olishda xatolik: {str(extract_error)}")
                        prompt = f"Quyidagi savol uchun o'zbek tilida aniq va qisqa javob ber: {user_input}"
                else:
                    prompt = f"Quyidagi savol uchun o'zbek tilida aniq va qisqa javob ber: {user_input}"
                
                direct_response = llm.predict(prompt)
                print(f"\nJavob: {direct_response}")
            except Exception as inner_e:
                print(f"To'g'ridan-to'g'ri javob olishda ham xatolik: {str(inner_e)}")

if __name__ == "__main__":
    main()
