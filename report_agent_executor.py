from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_community.chat_models import ChatOllama
from typing import AsyncGenerator
import asyncio
# Toollarni import qilish
from tools_llm.transport4.transport4_tool import Transport4Tool
from retriever.langchain_chroma import CustomEmbeddingFunction

# LLM modelini yaratish
llm = ChatOllama(
    model="mistral-small3.1:24b",
    base_url="http://localhost:11434",
    temperature=0.7,
    streaming=True  # Stream rejimini yoqish
)

# Bitta embedding modelini yaratish, barcha toollar uchun
shared_embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')

# Toollarni yaratish
transport4_tool = Transport4Tool("tools_llm/transport4/transport4.json", use_embeddings=True, embedding_model=shared_embedding_model)

# Embedding ma'lumotlarini tayyorlash
print("Barcha toollar uchun embedding ma'lumotlari tayyorlanmoqda...")
transport4_tool._prepare_embedding_data()

# System message yaratish
system_message = SystemMessage(content="""
        Siz O'zbekiston Davlat statistika qo'mitasi transport hisobotlari ma'lumotlari bilan ishlaydigan agentsiz. 
                    
        1. transport4_tool - 4-transport shakli hisoboti ma'lumotlarini qidirish va tahlil qilish uchun vosita.
                            
        QOIDALAR: 
        1. FAQAT O'ZBEK TILIDA javob bering
        2. Tooldan qaytgan inglizcha kalit so'zlarni o'zbek tiliga o'giring: NAME->nomi, CODE->kodi, DESCRIPTION->tavsifi, SECTION_TYPE->bo'lim turi, status->holat, count->soni, results->natijalar, message->xabar
        3. Natijalarni quyidagi formatda ko'rsating: "Men [qator]-satr, [ustun]-ustun ma'lumotlarini topdim. Nomi: [nomi], Kodi: [kodi], Tavsifi: [tavsifi], Bo'lim turi: [bo'lim turi]"
        4. Hech qachon ingliz tilida yoki inglizcha kalit so'zlarni ko'rsatmang
        5. Final Answer ham faqat o'zbek tilida bo'lishi kerak
    """)

# Barcha toollarni bir agent ichida birlashtirish
report_agent = initialize_agent(
    tools=[transport4_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output"),
    system_message=system_message
)

# Stream javob qaytaruvchi funksiya
async def agent_stream(query: str) -> AsyncGenerator[str, None]:
    """
    Agent javobini stream ko'rinishida qaytaradi
    
    Args:
        query: Foydalanuvchi so'rovi
        
    Returns:
        AsyncGenerator: Stream ko'rinishidagi javob
    """
    response = ""
    async for chunk in report_agent.astream({"input": query}):
        if "output" in chunk:
            chunk_text = chunk["output"]
            response += chunk_text
            yield chunk_text

# Agentni export qilish
__all__ = ['report_agent', 'agent_stream']

async def main():
    print("O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun chatbot")
    print("Chiqish uchun 'exit', 'quit' yoki 'chiqish' deb yozing")
    print("-" * 50)
    
    while True:
        user_input = input("User: ").strip().lower()
        if user_input in ['exit', 'quit', 'chiqish']:
            break
        
        async for chunk in agent_stream(user_input):
            print(chunk, end="")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())