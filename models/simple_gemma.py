# Use a pipeline as a high-level helper
from transformers import pipeline

# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")
model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-27b-it")

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("image-text-to-text", model=model, tokenizer=processor)
pipe(messages)