from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from typing import AsyncGenerator

# Umumiy modellarni import qilish
from shared_models import llm, shared_embedding_model

# Toollarni import qilish
from tools_llm.dbibt.dbibt_tool import DBIBTTool
from tools_llm.soato.soato_tool import SoatoTool
from tools_llm.nation.nation_tool import NationTool
from tools_llm.ckp.ckp_tool_simple import CkpTool
from tools_llm.country.country_tool import CountryTool
from tools_llm.thsh.thsh_tool import THSHTool
from tools_llm.tif_tn.tif_tn_tool import TIFTNTool

# Toollarni yaratish
soato_tool = SoatoTool("tools_llm/soato/soato.json", use_embeddings=True, embedding_model=shared_embedding_model)
nation_tool = NationTool("tools_llm/nation/nation_data.json", use_embeddings=True, embedding_model=shared_embedding_model)
ckp_tool = CkpTool("tools_llm/ckp/ckp.json", use_embeddings=True, embedding_model=shared_embedding_model)
dbibt_tool = DBIBTTool("tools_llm/dbibt/dbibt.json", use_embeddings=True, embedding_model=shared_embedding_model)
country_tool = CountryTool("tools_llm/country/country.json", use_embeddings=True, embedding_model=shared_embedding_model)
thsh_tool = THSHTool("tools_llm/thsh/thsh.json", use_embeddings=True, embedding_model=shared_embedding_model)
tif_tn_tool = TIFTNTool("tools_llm/tif_tn/tif_tn.json", use_embeddings=True, embedding_model=shared_embedding_model)

# Embedding ma'lumotlarini tayyorlash
print("Barcha toollar uchun embedding ma'lumotlari tayyorlanmoqda...")
soato_tool._prepare_embedding_data()
nation_tool._prepare_embedding_data()
ckp_tool._prepare_embedding_data()
dbibt_tool._prepare_embedding_data()
country_tool._prepare_embedding_data()
thsh_tool._prepare_embedding_data()
tif_tn_tool._prepare_embedding_data()

# System message yaratish
system_message = SystemMessage(content="""Siz O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun maxsus agentsiz. 
                    Sizda quyidagi toollar mavjud:
                    
                    1. soato_tool - O'zbekiston Respublikasining ma'muriy-hududiy birliklari (SOATO/MHOBIT) ma'lumotlarini qidirish uchun mo'ljallangan professional vosita. Bu tool LangChain best practices asosida yaratilgan va quyidagi imkoniyatlarni taqdim etadi:
                       - Viloyatlar, tumanlar, shaharlar va aholi punktlarning soato yoki mhobt kodlari va nomlari
                       - SOATO/MHOBIT kodlari bo'yicha aniq qidiruv
                       - Ma'muriy birliklarning ierarxik tuzilishi
                       - Viloyatlar, tumanlar va aholi punktlari soni haqida ma'lumotlar
                       Misol so'rovlar: "Toshkent viloyati tumanlari mhobt kodlari", "Samarqand viloyati soato si qanday", "1710 soato kodi", "Buxoro viloyati tumanlari soni", "Andijon ma'muriy markazi"
                    
                    2. nation_tool - O'zbekiston Respublikasi millat klassifikatori ma'lumotlarini qidirish uchun tool. Bu tool orqali millat kodi yoki millat nomi bo"yicha qidiruv qilish mumkin. Masalan: "01" (o'zbek), "05" (rus), yoki "tojik" kabi so"rovlar bilan qidiruv qilish mumkin. Tool millat kodi, nomi va boshqa tegishli ma'lumotlarni qaytaradi.
                    
                    3. ckp_tool - MST/CKP (Mahsulotlarning tasniflagichi) ma'lumotlarini qidirish va tahlil qilish uchun mo'ljallangan vosita. Bu tool orqali mahsulotlar kodlari, nomlari va tasniflarini izlash, ularning ma'lumotlarini ko"rish va tahlil qilish mumkin. Qidiruv so'z, kod yoki tasnif bo"yicha amalga oshirilishi mumkin.
                    
                    4. dbibt_tool - O'zbekiston Respublikasi Davlat va xo'jalik boshqaruvi idoralarini belgilash tizimi (DBIBT) ma'lumotlarini qidirish uchun tool. Bu tool orqali DBIBT kodi, tashkilot nomi, OKPO/KTUT yoki STIR/INN raqami bo"yicha qidiruv qilish mumkin. Masalan: "08824", "Vazirlar Mahkamasi", yoki "07474" kabi so"rovlar bilan qidiruv qilish mumkin.

                    5. country_tool - Davlatlar ma'lumotlarini qidirish uchun mo'ljallangan vosita. Bu tool orqali davlatlarning qisqa nomi, to'liq nomi, harf kodi va raqamli kodi bo'yicha qidiruv qilish mumkin. Misol uchun: "AQSH", "Rossiya", "UZ", "398" (Qozog'iston raqamli kodi) kabi so'rovlar orqali ma'lumotlarni izlash mumkin. Natijalar davlat kodi, qisqa nomi, to'liq nomi va kodlari bilan qaytariladi.

                    6. thsh_tool - Tashkiliy-huquqiy shakllar (THSH) ma'lumotlarini qidirish uchun mo'ljallangan vosita. Bu tool orqali tashkiliy-huquqiy shakllarning kodi, nomi, qisqa nomi va boshqa ma'lumotlar bo'yicha qidiruv qilish mumkin. Misol uchun: "110" (Xususiy korxona), "Aksiyadorlik jamiyati", "MJ" (Mas'uliyati cheklangan jamiyat) kabi so'rovlar orqali ma'lumotlarni izlash mumkin.
                    
                    7. tif_tn_tool - Tashqi iqtisodiy faoliyat tovarlar nomenklaturasi (TIF-TN). Bu tool orqali tashqi iqtisodiy faoliyat tovarlar nomenklaturasi bo'yicha qidiruv qilish mumkin. Misol uchun: '0102310000' (Tirik yirik shoxli qoramol), '0101210000' (Tirik otlar, eshaklar, xachirlar va xo' tiklar), '0102399000' (Tirik yirik shoxli qoramol) kabi so'rovlar orqali ma'lumotlarni izlash mumkin.
                    
                    Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning. 
                    Agar foydalanuvchi SOATO/MHOBIT (viloyat, tuman, shahar), millat, MST, davlat ma'lumotlari yoki tashkiliy-huquqiy shakllar haqida so'rasa, tegishli toolni chaqiring.
                    Toollarni chaqirish uchun Action formatidan foydalaning.
                    JUDA MUHIM QOIDALAR: 
                    1. Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning
                    2. Tool ishlatganingizdan keyin MAJBURIY ravishda Final Answer bering
                    """)

# Barcha toollarni bir agent ichida birlashtirish
combined_agent = initialize_agent(
    tools=[soato_tool, nation_tool, ckp_tool, dbibt_tool, country_tool, thsh_tool, tif_tn_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
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
    async for chunk in combined_agent.astream({"input": query}):
        if "output" in chunk:
            chunk_text = chunk["output"]
            response += chunk_text
            yield chunk_text

# Agentni export qilish
__all__ = ['combined_agent', 'agent_stream']