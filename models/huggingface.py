from transformers import pipeline

# Modelni yaratish
model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device="cuda"  # GPU bo'lsa "cuda", aks holda "cpu" ishlatiladi
)
