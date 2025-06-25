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
    model="gemma3:27b",
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
       "Men [bob]-bob, [ustun]-ustun ma'lumotlarini topdim. Bob nomi: [bob nomi], Ustun nomi: [ustun tavsifi], Bo'lim turi: [bo'lim_turi]"
    4. Agar nomi, tavsifi yoki description so'ralgan bo'lsa, natijada topilgan barcha ma'lumotlarni ko'rsating - bob nomi, ustun tavsifi, bo'lim turi, qat'iy nazoratlar va qat'iy bo'lmagan nazoratlar.
    5. Agar so'rov aniq bo'lmasa, foydalanuvchidan qo'shimcha ma'lumot so'rang.
    6. Ingliz tilidagi xabarlarni foydalanuvchiga ko'rsatmang.
    7. Ustun haqida so'ralgan bo'lsa (masalan "102-satr 1-ustun qanday nomalandi?"), javobda ustunning barcha ma'lumotlarini ko'rsating, faqat tavsifi bilangina cheklanmang.
    8. Barcha o'ylash jarayonlarini (Thought) ham faqat o'zbek tilida yozing. "Thought" o'rniga "Fikr:" deb yozing.
    9. Barcha harakat (Action) va kuzatuv (Observation) qismlarini ham o'zbek tilida yozing.
    10. MUHIM: Har qanday so'rovga javob berishdan oldin, ALBATTA transport4_tool ni ishlatib ma'lumotlarni qidiring. Hech qachon o'z bilimlaringizdan foydalanib javob bermang, chunki bu noto'g'ri natijalar berishi mumkin.
    11. Agar so'rovda bob yoki ustun haqida so'ralgan bo'lsa (masalan "3-bob 3-ustun qanday nomalandi?"), albatta transport4_tool ni ishlatib, shu bob va ustun ma'lumotlarini qidiring va topilgan ma'lumotlar asosida javob bering.
    12. JUDA MUHIM: Hech qachon "I'm sorry, but I don't have access..." kabi javoblarni bermang. Har qanday so'rovga javob berishda ALBATTA quyidagi formatda transport4_tool ni chaqiring:
        ```json
        {
          "action": "transport4_tool",
          "action_input": {
            "so_rov": "foydalanuvchi so'rovi"
          }
        }
        ```
    13. Agar so'rovda "description", "nomi" yoki "tavsifi" so'zlari bo'lsa, bu ustun yoki bob haqidagi barcha ma'lumotlar so'ralayotganini bildiradi. Bunday so'rovlarda albatta transport4_tool ni ishlatib, tegishli ma'lumotlarni qidiring va natijada topilgan barcha ma'lumotlarni ko'rsating.
    14. JUDA MUHIM: Qidiruv natijalarini diqqat bilan tahlil qiling. Agar so'ralgan satr yoki ustun raqami natijada qaytarilgan ma'lumotlarga mos kelmasa, buni foydalanuvchiga aniq aytib bering. Masalan, agar foydalanuvchi "203-satr" haqida so'rasa, lekin natijada "201-satr" qaytarilsa, "Kechirasiz, 203-satr topilmadi, lekin 201-satr haqida ma'lumot bor" deb javob bering.
    15. Agar qidiruv natijasi bo'sh bo'lsa (natijalar_yoq), foydalanuvchiga "Kechirasiz, so'ralgan ma'lumot topilmadi" deb javob bering.
    16. JUDA MUHIM: Har qanday holatda ham FAQAT O'ZBEK TILIDA javob bering. Hatto "Final Answer" ham o'zbek tilida bo'lishi kerak. Ingliz tilidagi so'zlarni o'zbek tiliga o'giring.
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