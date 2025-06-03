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
from tools_llm.ckp.ckp_tool_simple import CkpTool
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
    
    # CKP Tool ni yaratish
    print("CKP Tool yuklanmoqda...")
    ckp_simple = CkpTool()
    
    # CKP Tool ni Langchain Tool sifatida o'rash
    ckp_tool = Tool(
        name="ckp_tool",
        description="MST (Mahsulotlarning statistik tasniflagichi) ma'lumotlarini qidirish uchun tool",
        func=ckp_simple.run
    )
    print("CKP Tool yuklandi.")
    
    # Suhbat xotirasini ishga tushirish
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Agentni CKP Tool bilan ishga tushirish
    agent = initialize_agent(
        tools=[ckp_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    print("\nMST (Mahsulotlarning statistik tasniflagichi) ma'lumotlarini qidirish tizimi.")
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
            # Tool._run() o'rniga to'g'ridan-to'g'ri CkpTool.run() metodini ishlatamiz
            response = agent.run(user_input)  # LangChain agenti orqali so'rovni yuborish
            # response = ckp_simple.run(user_input)  # To'g'ridan-to'g'ri CKP Tool ga so'rovni yuborish
            print(f"\nJavob:\n{response}")
        except Exception as e:
            print(f"Xatolik yuz berdi: {str(e)}")

if __name__ == "__main__":
    main()
