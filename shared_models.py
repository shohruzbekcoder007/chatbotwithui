"""
Umumiy modellar va embedding funksiyalari uchun modul.
Barcha agentlar uchun bitta model va embedding funksiyasini ta'minlaydi.
"""
from langchain_community.chat_models import ChatOllama
from retriever.langchain_chroma import CustomEmbeddingFunction

# LLM modelini yaratish - barcha agentlar uchun umumiy
llm = ChatOllama(
    model="gemma3:27b",
    base_url="http://localhost:11434",
    temperature=0.3,  # Aniqroq javoblar uchun temperature ni kamaytiramiz
    streaming=True
)

# Embedding modelini yaratish - barcha toollar uchun umumiy
shared_embedding_model = CustomEmbeddingFunction(model_name='BAAI/bge-m3')

# Modellarni export qilish
__all__ = ['llm', 'shared_embedding_model']
