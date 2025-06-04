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
from tools_llm.nation.nation_tool import NationTool
from langchain.tools import Tool

def main():
    # Ollama LLM ni ishga tushurish
    llm = OllamaLLM(
        base_url="http://localhost:11434",  # Standart Ollama URL
        model="devstral",  # Ollamadan mavjud model
        temperature=0.7
    )
    
    # CUDA mavjudligini tekshirish (GPU tezlatish uchun)
    use_gpu = torch.cuda.is_available()
    print(f"GPU mavjud: {use_gpu}")
    
    # Nation Tool ni yaratish
    print("Nation Tool yuklanmoqda...")
    nation_tool_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_llm", "nation", "nation_data.json")
    nation_tool = NationTool(nation_file_path=nation_tool_path)
    
    # Tool ni LangChain formatiga o'tkazish
    nation_langchain_tool = Tool(
        name="nation_tool",
        func=nation_tool._run,
        description="Millatlar klassifikatori."
    )
    
    # Suhbat xotirasini yaratish
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Agentni yaratish
    agent = initialize_agent(
        [nation_langchain_tool],
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Testlash
    print("\nTest so'rovlar:")
    test_queries = [
        "Rus millati kodi nima?",
        "30 kodi qaysi millatga tegishli?",
        "O'zbek millati kodi qanday?",
        "Osiyo"
    ]
    
    for query in test_queries:
        print(f"\nSo'rov: {query}")
        try:
            result = agent.run(query)
            print(f"Natija: {result}")
        except Exception as e:
            print(f"Xatolik: {str(e)}")

if __name__ == "__main__":
    main()