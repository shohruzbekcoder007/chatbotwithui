import os
import sys
import torch
from langchain.agents import AgentType, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

# Add the tools_llm directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the SoatoTool
from tools_llm.soato.soato_tool import SoatoTool

def main():
    # Initialize the Ollama LLM
    llm = OllamaLLM(
        base_url="http://localhost:11434",  # Default Ollama URL
        model="llama3.1:8b",  # Using an available model from Ollama
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

if __name__ == "__main__":
    main()
