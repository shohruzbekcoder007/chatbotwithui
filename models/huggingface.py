from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GemmaModel:
    def __init__(self, model_name="google/gemma-2b"):
        """Modelni va tokenizerni yuklab olish"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, force_download=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            force_download=True
        )
    
    def generate_text(self, prompt, context="", max_length=512, temperature=0.7):
        """
        Ma’lumot asosida generatsiya qilish.

        Args:
            prompt (str): Foydalanuvchi so‘rovi
            context (str): Qo‘shimcha ma’lumot
            max_length (int): Maksimal token uzunligi
            temperature (float): Generatsiya randomness darajasi

        Returns:
            str: Model javobi
        """
        # Foydalanuvchi so‘roviga kontekstni qo‘shish
        full_prompt = f"Ma'lumot:\n{context}\n\nSavol:\n{prompt}\n\nJavob:"

        # Tokenizatsiya
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        # Generatsiya qilish
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_length,  # ⬅️ Faqat yangi 512 token generatsiya qiladi
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Matnni dekod qilish
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Modelni yaratish
model = GemmaModel()

# # Model ishlatish misoli
# if __name__ == "__main__":
#     # Ma'lumot berish
#     context_info = """
#     O'zbekiston Markaziy Osiyodagi davlat bo‘lib, poytaxti Toshkent shahridir.
#     Uning aholisi 36 milliondan ortiq. Rasmiy tili o‘zbek tili. 
#     Asosiy valyutasi so‘m (UZS). Iqtisodiyoti paxta yetishtirish, sanoat va gaz ishlab chiqarishga asoslangan.
#     """

#     # Foydalanuvchi savoli
#     prompt = "O'zbekistonning poytaxti qaysi shahar?"

#     # Generatsiya qilish
#     response = model.generate_text(prompt, context=context_info)
#     print("\nJavob:", response)

