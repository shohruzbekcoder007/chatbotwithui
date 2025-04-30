from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
from dotenv import load_dotenv

# .env faylini yuklash
load_dotenv()

# Model identifikatori va saqlash yo'li
model_id = "google/gemma-7b-it"  # 27b o'rniga 7b modelini ishlatamiz, chunki u ochiq foydalanish uchun mavjud
local_dir = "local_model"

# Hugging Face tokenini environment variable'dan olish
token = os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set")

# 4bit konfiguratsiyasi (VRAMni tejash uchun)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit yuklash
    bnb_4bit_use_double_quant=True,         # Ikki bosqichli kvantlash
    bnb_4bit_quant_type="nf4",              # NF4 kvantlash turi
    bnb_4bit_compute_dtype=torch.float16    # Hisoblashda float16 ishlatish
)

# Tokenizerni yuklash va saqlash
print(" Tokenizer yuklanmoqda...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=token,
    trust_remote_code=True
)
tokenizer.save_pretrained(local_dir)

# Modelni yuklash va saqlash (4bit, VRAM uchun optimallashtirilgan)
print(" Model yuklanmoqda...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",               # Avtomatik GPU/CPU taqsimoti
    quantization_config=bnb_config,  # 4-bit config
    torch_dtype=torch.float16,       # GPU uchun yengil format
    token=token                      # Hugging Face token
)
model.save_pretrained(local_dir)

print(" Model va tokenizer saqlandi:", local_dir)

# Test: local_model dan yuklab, javob olish
print(" Local modeldan foydalanilmoqda...")
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "O'zbekiston Respublikasi Prezidenti kim?"
response = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)

print(" Javob:\n", response[0]["generated_text"])
