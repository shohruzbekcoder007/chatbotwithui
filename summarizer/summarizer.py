import os
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from typing import Optional, List, Dict, Any, Union
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextSummarizer:
    """
    Text summarizer using Google's MT5-xl model.
    The model will be saved to and loaded from ./summarizer_model directory.
    Supports summarizing individual texts and lists of text chunks with subsequent combination.
    """
    
    def __init__(self, model_name: str = "google/mt5-xl", model_dir: str = "./summarizer_model"):
        """
        Initialize the text summarizer.
        
        Args:
            model_name: The name of the model to use from Hugging Face.
            model_dir: The directory to save and load the model from.
        """
        self.model_name = model_name
        self.model_dir = model_dir
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Check if model exists in local directory
        if self._is_model_saved():
            logger.info(f"Loading model from local directory: {self.model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained(self.model_dir)
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_dir)
        else:
            logger.info(f"Downloading model {self.model_name} from Hugging Face")
            self.tokenizer = MT5Tokenizer.from_pretrained(self.model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Save model to local directory
            logger.info(f"Saving model to {self.model_dir}")
            self.tokenizer.save_pretrained(self.model_dir)
            self.model.save_pretrained(self.model_dir)
        
        # Move model to GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
    
    def _is_model_saved(self) -> bool:
        """Check if the model is saved in the local directory."""
        tokenizer_files = ["tokenizer_config.json", "special_tokens_map.json", "spiece.model"]
        model_files = ["pytorch_model.bin", "config.json"]
        
        for file in tokenizer_files + model_files:
            if not os.path.exists(os.path.join(self.model_dir, file)):
                return False
        return True
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, 
                  num_beams: int = 4, prefix: str = "summarize: ") -> str:
        """
        Summarize the input text.
        
        Args:
            text: The text to summarize.
            max_length: Maximum length of the summary.
            min_length: Minimum length of the summary.
            num_beams: Number of beams for beam search.
            prefix: Prefix to add before the input text to guide the model.
            
        Returns:
            The summarized text.
        """
        # Prepare input text with prefix
        input_text = prefix + text
        
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def batch_summarize(self, texts: List[str], max_length: int = 150, min_length: int = 40,
                         num_beams: int = 4, prefix: str = "summarize: ") -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize.
            max_length: Maximum length of the summaries.
            min_length: Minimum length of the summaries.
            num_beams: Number of beams for beam search.
            prefix: Prefix to add before each input text to guide the model.
            
        Returns:
            List of summarized texts.
        """
        # Add prefix to each text
        input_texts = [prefix + text for text in texts]
        
        # Tokenize input texts
        batch_inputs = self.tokenizer(input_texts, return_tensors="pt", max_length=1024, 
                                     truncation=True, padding=True)
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        
        # Generate summaries
        with torch.no_grad():
            batch_summary_ids = self.model.generate(
                **batch_inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode the summaries
        batch_summaries = [self.tokenizer.decode(ids, skip_special_tokens=True) 
                          for ids in batch_summary_ids]
        
        return batch_summaries
        
    async def summarize_chunks(self, chunks: List[str], max_chunk_length: int = 150, min_chunk_length: int = 40,
                           max_combined_length: int = 300, min_combined_length: int = 100,
                           num_beams: int = 4, prefix: str = "summarize: ") -> str:
        """
        Summarize each chunk in the list, then combine all summaries into a final summary.
        
        Args:
            chunks: List of text chunks to summarize.
            max_chunk_length: Maximum length for each chunk summary.
            min_chunk_length: Minimum length for each chunk summary.
            max_combined_length: Maximum length for the final combined summary.
            min_combined_length: Minimum length for the final combined summary.
            num_beams: Number of beams for beam search.
            prefix: Prefix to add before each input text to guide the model.
            
        Returns:
            Final combined summary of all chunks.
        """
        if not chunks:
            return ""
            
        # First, get summaries for each chunk in parallel
        chunk_summaries = self.batch_summarize(
            chunks,
            max_length=max_chunk_length,
            min_length=min_chunk_length,
            num_beams=num_beams,
            prefix=prefix
        )
        
        # Then combine all summaries into one text
        combined_text = "\n".join(chunk_summaries)
        
        # Finally, summarize the combined text to get the final summary
        final_summary = self.summarize(
            combined_text,
            max_length=max_combined_length,
            min_length=min_combined_length,
            num_beams=num_beams,
            prefix=prefix
        )
        
        return final_summary
        
    async def process_text_chunks(self, chunks_list: List[str], **kwargs) -> str:
        """
        Process a list of text chunks, summarize each one, and combine them.
        This is a convenience method that runs the async summarize_chunks method.
        
        Args:
            chunks_list: List of text chunks to process.
            **kwargs: Additional parameters to pass to summarize_chunks.
            
        Returns:
            Combined summary of all text chunks.
        """
        # Run the async method in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(await self.summarize_chunks(chunks_list, **kwargs))
        finally:
            loop.close()

summarizer = TextSummarizer()

# Example usage
# if __name__ == "__main__":
#     # Initialize the summarizer
#     summarizer = TextSummarizer()
    
#     # Example 1: Summarize a single text
#     text = """
#     Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans. 
#     AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
#     The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
#     This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
#     """
    
#     # Generate summary for single text
#     summary = summarizer.summarize(text)
#     print("Single text summary:\n", summary)
    
#     # Example 2: Process and summarize a list of text chunks
#     chunks = [
#         "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals and humans.",
#         "AI applications include advanced web search engines, recommendation systems (used by YouTube, Amazon and Netflix), understanding human speech (such as Siri and Alexa).",
#         "Self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), automated decision-making, and competing at the highest level in strategic game systems (such as chess and Go).",
#         "As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the definition of AI, a phenomenon known as the AI effect."
#     ]
    
#     # Process the chunks
#     combined_summary = summarizer.process_text_chunks(chunks)
#     print("\nCombined summary from chunks:\n", combined_summary)