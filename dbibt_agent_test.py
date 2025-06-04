import os
import sys
import re
import torch
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import Tool

# Asosiy loyiha katalogini import path ga qo'shish
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DBIBT Tool ni import qilish
from tools_llm.dbibt.dbibt_tool import DBIBTTool

def main():
    # Ollama LLM ni ishga tushirish
    llm = OllamaLLM(
        base_url="http://localhost:11434",  # Standart Ollama URL
        model="devstral",  # Ollamadan mavjud model
        temperature=0.7
    )
    
    # CUDA mavjudligini tekshirish (GPU tezlatish uchun)
    use_gpu = torch.cuda.is_available()
    print(f"GPU mavjud: {use_gpu}")
    
    # DBIBT Tool ni yaratish
    print("DBIBT Tool yuklanmoqda...")
    dbibt_tool = DBIBTTool("tools_llm/dbibt/dbibt.json", use_embeddings=True)
    
    # DBIBT Tool ni Langchain Tool sifatida o'rash
    dbibt_tool_wrapped = Tool(
        name="dbibt_tool",
        description="DBIBT ma'lumotlarini qidirish uchun tool. Bu tool orqali DBIBT kodlari, OKPO, INN va tashkilot nomlari bo'yicha qidiruv qilish mumkin.",
        func=dbibt_tool._run
    )
    print("DBIBT Tool yuklandi.")
    
    # Suhbat xotirasini ishga tushirish
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Agentni DBIBT Tool bilan ishga tushirish
    agent = initialize_agent(
        tools=[dbibt_tool_wrapped],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    print("\nDBIBT (DBIBT ma'lumotlarini qidirish tizimi.")
    print("Chiqish uchun 'exit' yoki 'quit' deb yozing.")
    print("To'g'ridan-to'g'ri qidiruv uchun 'direct:' prefiksini ishlating.")
    
    # Foydalanuvchi bilan suhbat tsikli
    while True:
        user_input = input("\nSavol: ")
        
        # Chiqish shartini tekshirish
        if user_input.lower() in ["exit", "quit", "chiqish"]:
            print("Dastur yakunlandi.")
            break
        
        # Agentga so'rovni yuborish
        try:
            # To'g'ridan-to'g'ri qidiruv
            if user_input.startswith("direct:"):
                query = user_input[7:].strip()  
                response = dbibt_tool._run(query)
                print(f"\nJavob (to'g'ridan-to'g'ri qidiruv):\n{response}")
            else:
                # LangChain agenti orqali so'rovni yuborish
                response = agent.run(user_input)
                print(f"\nJavob:\n{response}")
        except Exception as e:
            print(f"Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    main()
