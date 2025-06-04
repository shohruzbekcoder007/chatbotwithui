from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_community.chat_models import ChatOllama
import time

# Toollarni import qilish
from tools_llm.dbibt.dbibt_tool import DBIBTTool
from tools_llm.soato.soato_tool import SoatoTool
from tools_llm.nation.nation_tool import NationTool
from tools_llm.ckp.ckp_tool_simple import CkpTool
from retriever.langchain_chroma import CustomEmbeddingFunction

# LLM modelini yaratish
llm = ChatOllama(
    model="devstral",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Bitta embedding modelini yaratish, barcha toollar uchun
shared_embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')

# Toollarni yaratish
soato_tool = SoatoTool("tools_llm/soato/soato.json", use_embeddings=True, embedding_model=shared_embedding_model)
nation_tool = NationTool("tools_llm/nation/nation_data.json", use_embeddings=True, embedding_model=shared_embedding_model)
ckp_tool = CkpTool("tools_llm/ckp/ckp.json", use_embeddings=True, embedding_model=shared_embedding_model)
dbibt_tool = DBIBTTool("tools_llm/dbibt/dbibt.json", use_embeddings=True, embedding_model=shared_embedding_model)

# Embedding ma'lumotlarini tayyorlash
print("Barcha toollar uchun embedding ma'lumotlari tayyorlanmoqda...")
soato_tool._prepare_embedding_data()
nation_tool._prepare_embedding_data()
ckp_tool._prepare_embedding_data()
dbibt_tool._prepare_embedding_data()

# Barcha toollarni bir agent ichida birlashtirish
combined_agent = initialize_agent(
    tools=[soato_tool, nation_tool, ckp_tool, dbibt_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    system_message=SystemMessage(content="""Siz O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun maxsus agentsiz. 
        Sizda quyidagi toollar mavjud:
        1. soato_tool - O"zbekiston Respublikasining ma"muriy-hududiy birliklari (SOATO/MHOBIT) ma"lumotlarini qidirish uchun mo"ljallangan vosita. Bu tool orqali viloyatlar, tumanlar, shaharlar va boshqa ma"muriy birliklarning kodlari, nomlari va joylashuvlarini topish mumkin. Misol uchun: "Toshkent shahar", "Samarqand viloyati", "1703" (Namangan viloyati kodi), "Buxoro tumani" kabi so"rovlar orqali ma"lumotlarni izlash mumkin.
        
        2. nation_tool - O"zbekiston Respublikasi millat klassifikatori ma"lumotlarini qidirish uchun tool. Bu tool orqali millat kodi yoki millat nomi bo"yicha qidiruv qilish mumkin. Masalan: "01" (o"zbek), "05" (rus), yoki "tojik" kabi so"rovlar bilan qidiruv qilish mumkin. Tool millat kodi, nomi va boshqa tegishli ma"lumotlarni qaytaradi.
        
        3. ckp_tool - MST/CKP (Mahsulotlarning statistik tasniflagichi) ma"lumotlarini qidirish va tahlil qilish uchun mo"ljallangan vosita. Bu tool orqali mahsulotlar kodlari, nomlari va tasniflarini izlash, ularning ma"lumotlarini ko"rish va tahlil qilish mumkin. Qidiruv so"z, kod yoki tasnif bo"yicha amalga oshirilishi mumkin.
        
        4. dbibt_tool - O"zbekiston Respublikasi Davlat va xo"jalik boshqaruvi idoralarini belgilash tizimi (DBIBT) ma"lumotlarini qidirish uchun tool. Bu tool orqali DBIBT kodi, tashkilot nomi, OKPO/KTUT yoki STIR/INN raqami bo"yicha qidiruv qilish mumkin. Masalan: "08824", "Vazirlar Mahkamasi", yoki "07474" kabi so"rovlar bilan qidiruv qilish mumkin.

        Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning. 
        Agar foydalanuvchi SOATO/MHOBIT, millat yoki MST ma'lumotlari haqida so'rasa, tegishli toolni chaqiring.
        Toollarni chaqirish uchun Action formatidan foydalaning.""")
    )


# from models_llm.langchain_ollamaCustom import get_model_instance
# model = get_model_instance(combined_agent=combined_agent)

# if __name__ == "__main__":
#     print("O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun chatbot")
#     print("Chiqish uchun 'exit', 'quit' yoki 'chiqish' deb yozing")
#     print("-" * 50)
    
#     while True:
#         try:
#             # Foydalanuvchidan savolni qabul qilish
#             user_input = input("\nSavolingizni kiriting: ")
            
#             # Chiqish uchun tekshirish
#             if user_input.lower() in ["exit", "quit", "chiqish"]:
#                 print("Chatbot ishini tugatdi. Xayr!")
#                 break
            
#             # Bo'sh kiritishni tekshirish
#             if not user_input.strip():
#                 print("Iltimos, savol kiriting.")
#                 continue
            
#             print("\nJavob tayyorlanmoqda...")
            
#             # Vaqtni o'lchash boshlash
#             start_time = time.time()
            
#             # Agentni chaqirish
#             result = combined_agent.invoke({"input": user_input})
            
#             # Vaqtni o'lchash yakunlash
#             end_time = time.time()
#             execution_time = end_time - start_time
            
#             print("\nNatija:", result["output"])
#             print(f"Ishlash vaqti: {execution_time:.2f} soniya")
            
#         except KeyboardInterrupt:
#             print("\nChatbot ishini tugatdi. Xayr!")
#             break
#         except Exception as e:
#             print(f"\nXatolik yuz berdi: {str(e)}")
#             print("Xatolik tafsilotlari:", e.__class__.__name__)
