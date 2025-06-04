import os
import sys
import re
import torch
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory

# Asosiy loyiha katalogini import path ga qo'shish
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CKP Tool ni import qilish
from tools_llm.dbibt.dbibt_tool import DBIBTTool
from langchain.tools import Tool

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
    dbibt_tool = DBIBTTool()
    
    # DBIBT Tool ni Langchain Tool sifatida o'rash
    dbibt_tool = Tool(
        name="dbibt_tool",
        description="DBIBT (DBIBT ma'lumotlarini qidirish uchun tool",
        func=dbibt_tool.run
    )
    print("DBIBT Tool yuklandi.")
    
    # Suhbat xotirasini ishga tushirish
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Agentni DBIBT Tool bilan ishga tushirish
    agent = initialize_agent(
        tools=[dbibt_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    print("\nDBIBT (DBIBT ma'lumotlarini qidirish tizimi.")
    print("Chiqish uchun 'exit' yoki 'quit' deb yozing.")
    
    # Foydalanuvchi bilan suhbat tsikli
    while True:
        user_input = input("\nSavol: ")
        
        # Chiqish shartini tekshirish
        if user_input.lower() in ["exit", "quit", "chiqish"]:
            print("Dastur yakunlandi.")
            break
        
        # Agentga so'rovni yuborish
        try:
            # Tool._run() o'rniga to'g'ridan-to'g'ri DBIBTTool.run() metodini ishlatamiz
            response = agent.run(user_input)  # LangChain agenti orqali so'rovni yuborish
            # response = dbibt_tool.run(user_input)  # To'g'ridan-to'g'ri DBIBT Tool ga so'rovni yuborish
            print(f"\nJavob:\n{response}")
        except Exception as e:
            print(f"Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    main()
