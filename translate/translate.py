from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re

class Translator:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"
        self.local_model_path = os.path.join("local_models", self.model_name.replace("/", "_"))
        self.executor = ThreadPoolExecutor()
        self.max_chunk_length = 450  # Maximum safe length for translation
        
        # Check if model exists locally, if not download it
        if not os.path.exists(self.local_model_path):
            print("Downloading model for the first time...")
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Save model and tokenizer locally
            model.save_pretrained(self.local_model_path)
            tokenizer.save_pretrained(self.local_model_path)
            print(f"Model saved to {self.local_model_path}")
        else:
            print("Loading model from local directory...")
            
        # Load model and tokenizer from local directory
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.local_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        
        # Create translation pipeline
        self.pipeline = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang="rus_Cyrl",
            tgt_lang="uzn_Latn",
        )

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences"""
        # Split by common sentence endings while preserving them
        sentences = re.split(r'([.!?])', text)
        # Combine sentence endings with their sentences
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
            else:
                result.append(sentences[i])
        return [s.strip() for s in result if s.strip()]

    def _combine_sentences_into_chunks(self, sentences: list, max_length: int) -> list:
        """Combine sentences into chunks that don't exceed max_length"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _russian_to_uzbek_sync(self, text: str) -> str:
        """Synchronous function for Russian to Uzbek translation"""
        # Split text into sentences and then combine into manageable chunks
        sentences = self._split_into_sentences(text)
        chunks = self._combine_sentences_into_chunks(sentences, self.max_chunk_length)
        
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            result = self.pipeline(
                chunk,
                src_lang="rus_Cyrl",
                tgt_lang="uzn_Latn",
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            translated_chunks.append(result[0]['translation_text'])
        
        # Combine translated chunks
        return " ".join(translated_chunks)
    
    def _uzbek_to_russian_sync(self, text: str) -> str:
        """Synchronous function for Uzbek to Russian translation"""
        # Split text into sentences and then combine into manageable chunks
        sentences = self._split_into_sentences(text)
        chunks = self._combine_sentences_into_chunks(sentences, self.max_chunk_length)
        
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            result = self.pipeline(
                chunk,
                src_lang="uzn_Latn",
                tgt_lang="rus_Cyrl",
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            translated_chunks.append(result[0]['translation_text'])
        
        # Combine translated chunks
        return " ".join(translated_chunks)
    
    async def russian_to_uzbek(self, text: str) -> str:
        """Asynchronously translate Russian text to Uzbek"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._russian_to_uzbek_sync, text)
        return result
    
    async def uzbek_to_russian(self, text: str) -> str:
        """Asynchronously translate Uzbek text to Russian"""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._uzbek_to_russian_sync, text)
        return result

translator = Translator()

# Example usage:
async def main():
    # Russian to Uzbek example (long text)
    russian_text = """Статья 66. """
    
    print(f"Russian text:\n{russian_text}\n")
    result = await translator.russian_to_uzbek(russian_text)
    print(f"Uzbek translation:\n{result}\n")
    
    # Uzbek to Russian example (long text)
    uzbek_text = """66-modda."""
    
    print(f"Uzbek text:\n{uzbek_text}\n")
    result = await translator.uzbek_to_russian(uzbek_text)
    print(f"Russian translation:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())