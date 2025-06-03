from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_community.chat_models import ChatOllama

# Toollarni import qilish
from tools_llm.soato.soato_tool import SoatoTool
from tools_llm.nation.nation_tool import NationTool
from tools_llm.ckp.ckp_tool_simple import CkpTool

# LLM modelini yaratish
llm = ChatOllama(
    model="llama3.3",
    base_url="http://localhost:11434",
    temperature=0.7
)

# Toollarni yaratish
soato_tool = SoatoTool("tools_llm/soato/soato.json", use_embeddings=True)
nation_tool = NationTool("tools_llm/nation/nation_data.json", use_embeddings=True)
ckp_tool = CkpTool()

# Barcha toollarni bir agent ichida birlashtirish
combined_agent = initialize_agent(
    tools=[soato_tool, nation_tool, ckp_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    system_message=SystemMessage(content="""Siz O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun maxsus agentsiz. 
        Sizda quyidagi toollar mavjud:
        1. soato_tool - SOATO (ma'muriy-hududiy birliklar) ma'lumotlarini qidirish uchun
        2. nation_tool - millat klassifikatori ma'lumotlarini qidirish uchun
        3. ckp_tool - MST (mahsulotlarning statistik tasniflagichi) ma'lumotlarini qidirish uchun

        Foydalanuvchi so'roviga javob berish uchun ALBATTA ushbu toollardan foydalaning. 
        Agar foydalanuvchi SOATO, millat yoki MST ma'lumotlari haqida so'rasa, tegishli toolni chaqiring.
        Toollarni chaqirish uchun Action formatidan foydalaning.""")
    )


if __name__ == "__main__":
    print("O'zbekiston Respublikasi Davlat statistika qo'mitasi ma'lumotlari bilan ishlash uchun chatbot")
    print("Chiqish uchun 'exit', 'quit' yoki 'chiqish' deb yozing")
    print("-" * 50)
    
    while True:
        try:
            # Foydalanuvchidan savolni qabul qilish
            user_input = input("\nSavolingizni kiriting: ")
            
            # Chiqish uchun tekshirish
            if user_input.lower() in ["exit", "quit", "chiqish"]:
                print("Chatbot ishini tugatdi. Xayr!")
                break
            
            # Bo'sh kiritishni tekshirish
            if not user_input.strip():
                print("Iltimos, savol kiriting.")
                continue
            
            print("\nJavob tayyorlanmoqda...")
            # Agentni chaqirish
            result = combined_agent.invoke({"input": user_input})
            print("\nNatija:", result["output"])
            
        except KeyboardInterrupt:
            print("\nChatbot ishini tugatdi. Xayr!")
            break
        except Exception as e:
            print(f"\nXatolik yuz berdi: {str(e)}")
            print("Xatolik tafsilotlari:", e.__class__.__name__)

