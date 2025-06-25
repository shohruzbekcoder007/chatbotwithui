# main.py
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
    streaming=True
)

# Embedding modelini yaratish
shared_embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')

# Toollarni yaratish
transport4_tool = Transport4Tool(
    transport4_file_path="tools_llm/transport4/transport4.json",
    use_embeddings=True,
    embedding_model=shared_embedding_model
)

# Embedding ma'lumotlarini tayyorlash
print("Barcha toollar uchun embedding ma'lumotlari tayyorlanmoqda...")
transport4_tool._prepare_embedding_data()

# System message
system_message = SystemMessage(content="""
    Siz O'zbekiston Davlat statistika qo'mitasi transport hisobotlari ma'lumotlari bilan ishlaydigan agentsiz. 
    FAQAT O'ZBEK TILIDA JAVOB BERING. Hech qanday holatda ingliz tilida so'z yoki ibora ishlatmang.
    
    Quyidagi vositalar mavjud:
    1. transport4_tool - 4-transport shakli hisoboti ma'lumotlarini qidirish uchun.
    
    QOIDALAR:
    1. Barcha javoblar faqat o'zbek tilida bo'lishi kerak.
    2. Inglizcha kalit so'zlarni o'zbek tiliga o'giring:
       - NAME -> nomi
       - CODE -> kodi
       - DESCRIPTION -> tavsifi
       - SECTION_TYPE -> bo'lim_turi
       - status -> holat
       - count -> soni
       - results -> natijalar
       - message -> xabar
       - Thought -> Fikr
       - Action -> Harakat
       - Observation -> Kuzatuv
       - Final Answer -> Yakuniy Javob
    3. Natijalarni quyidagi formatda ko'rsating:
       "Men [qator]-qator, [ustun]-ustun ma'lumotlarini topdim. Nomi: [tavsifi]"
    4. Agar ustun haqida so'ralgan bo'lsa, ustunning nomi sifatida 'tavsifi' maydonidagi ma'lumotni ko'rsating, 'nomi' maydonidagi ma'lumotni emas.
    5. Agar so'rov aniq bo'lmasa, foydalanuvchidan qo'shimcha ma'lumot so'rang.
    6. Ingliz tilidagi xabarlarni foydalanuvchiga ko'rsatmang.
    7. Ustun haqida so'ralgan bo'lsa (masalan "102-satr 1-ustun qanday nomalandi?"), javobda faqat ustunning tavsifini ko'rsating, bo'limning sarlavhasini emas.
    8. Barcha o'ylash jarayonlarini (Thought) ham faqat o'zbek tilida yozing. "Thought" o'rniga "Fikr:" deb yozing.
    9. Barcha harakat (Action) va kuzatuv (Observation) qismlarini ham o'zbek tilida yozing.
""")

# Agentni yaratish
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
async def agent_stream(so_rov: str) -> AsyncGenerator[str, None]:
    """Agent javobini stream ko'rinishida qaytaradi."""
    response = ""
    async for chunk in report_agent.astream({"input": so_rov}):
        if "output" in chunk:
            chunk_text = chunk["output"]
            # Ingliz tilidagi so'zlarni o'zbekchaga o'girish
            chunk_text = (chunk_text.replace("NAME", "nomi")
                         .replace("CODE", "kodi")
                         .replace("DESCRIPTION", "tavsifi")
                         .replace("SECTION_TYPE", "bo'lim_turi")
                         .replace("status", "holat")
                         .replace("count", "soni")
                         .replace("results", "natijalar")
                         .replace("message", "xabar"))
            response += chunk_text
            yield chunk_text

async def main():
    print("O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun chatbot")
    print("Chiqish uchun 'exit', 'quit' yoki 'chiqish' deb yozing")
    print("-" * 50)
    
    while True:
        user_input = input("Foydalanuvchi: ").strip().lower()
        if user_input in ['exit', 'quit', 'chiqish']:
            break
        
        async for chunk in agent_stream(user_input):
            print(chunk, end="")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())