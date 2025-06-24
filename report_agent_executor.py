from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_community.chat_models import ChatOllama
from typing import AsyncGenerator
import asyncio
# Toollarni import qilish
from tools_llm.transport4.transport4_tool import Transport4Tool
from tools_llm.dbibt.dbibt_tool import DBIBTTool
from retriever.langchain_chroma import CustomEmbeddingFunction

# LLM modelini yaratish
llm = ChatOllama(
    model="devstral",
    base_url="http://localhost:11434",
    temperature=0.7,
    streaming=True  # Stream rejimini yoqish
)

# Bitta embedding modelini yaratish, barcha toollar uchun
shared_embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')

# Toollarni yaratish
transport4_tool = Transport4Tool("tools_llm/transport4/transport4.json", use_embeddings=True, embedding_model=shared_embedding_model)
# dbibt_tool = DBIBTTool("tools_llm/dbibt/dbibt.json", use_embeddings=True, embedding_model=shared_embedding_model)

# Embedding ma'lumotlarini tayyorlash
print("Barcha toollar uchun embedding ma'lumotlari tayyorlanmoqda...")
transport4_tool._prepare_embedding_data()
# dbibt_tool._prepare_embedding_data()

# System message yaratish
system_message = SystemMessage(content="""Siz O'zbekiston Respublikasi Davlat statistika qo'mitasi hisobotlar ma'lumotlari bilan ishlash uchun maxsus agentsiz. 
                    Sizda quyidagi toollar mavjud:
                    
                    1. transport4_tool - 4-transport shakli hisoboti ma'lumotlarini qidirish va tahlil qilish uchun mo'ljallangan vosita. Bu tool orqali:
                       - Yuk va yo'lovchi tashish faoliyati ko'rsatkichlari
                       - Avtomobil transporti ishi bo'yicha ma'lumotlar
                       - Tashilgan yuklar hajmi, yuk aylanmasi, tashilgan yo'lovchilar va yo'lovchi aylanmasi
                       - Hisobot bo'limlari, qatorlari va ustunlari bo'yicha ma'lumotlar
                       - Mantiqiy nazorat qoidalari
                       Misol so'rovlar: "Tashilgan yuklar hajmi", "Yo'lovchi aylanmasi", "1-BOB ma'lumotlari", "101-satr nima haqida"
                    
                    Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning. 
                    Agar foydalanuvchi transport hisobotlari yoki tashkilotlar haqida so'rasa, tegishli toolni chaqiring.
                    Toollarni chaqirish uchun Action formatidan foydalaning.
                    JUDA MUHIM QOIDALAR: 
                    1. Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning
                    2. Tool ishlatganingizdan keyin MAJBURIY ravishda Final Answer bering
                    3. Tool qaytargan natijalarni to'liq va batafsil ko'rsating. Natijalar haqida umumiy gaplar bilan cheklanmang
                    4. Agar foydalanuvchi ma'lum bir qator va ustun haqida so'rasa, shu qator va ustunning aniq ma'lumotlarini ko'rsating
                    5. Har doim natijalarni o'zbek tilida tushunarli formatda taqdim eting
                    6. Faqat o'zbek tilida javob berishingiz kerak
                    7. JUDA MUHIM: Hech qachon ingliz tilida javob bermang, faqat va faqat o'zbek tilida javob bering
                    8. Agar tool ingliz tilidagi ma'lumotlarni qaytarsa, ularni o'zbek tiliga tarjima qilib, keyin foydalanuvchiga taqdim eting
                    9. Hatto tooldan qaytgan ma'lumotlar ingliz tilida bo'lsa ham, siz bu ma'lumotlarni o'zbek tiliga o'girib, o'zbek tilida javob bering
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