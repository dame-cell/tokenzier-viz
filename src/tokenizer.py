from transformers import AutoTokenizer
from typing import List, Dict, Union
import tiktoken
import time

class Tokenizer:
    def __init__(self, tokenizer_list, tiktoken_model):
        self.tokenizer_list = tokenizer_list  # List of Hugging Face tokenizers
        self.tiktoken_model = tiktoken_model  # Tiktoken model name
        self.time = time

    def format_time(self, elapsed_time):
        """Formats the elapsed time to a readable string."""
        if elapsed_time >= 3600:
            return f"{elapsed_time // 3600} hours {elapsed_time % 3600 // 60} minutes {elapsed_time % 60:.2f} seconds"
        elif elapsed_time >= 60:
            return f"{elapsed_time // 60} minutes {elapsed_time % 60:.2f} seconds"
        elif elapsed_time >= 1:
            return f"{elapsed_time:.2f} seconds"
        else:
            return f"{elapsed_time * 1000:.2f} milliseconds"

    def tokenize(self, text: str):
        if not isinstance(text, str):
            raise TypeError("Expected a string input.")
        
        start_time = self.time.time()

        results = {"tiktoken": [], "huggingface": {}}
        
        # Tokenize using Tiktoken (always used)
        try:
            tokenizer = tiktoken.encoding_for_model(self.tiktoken_model)
            results["tiktoken"] = tokenizer.encode(text)
        except Exception as e:
            print(f"Error with tiktoken tokenizer: {str(e)}")

        # Tokenize using Hugging Face tokenizers (always used)
        hf_tokenizers = [AutoTokenizer.from_pretrained(tok) for tok in self.tokenizer_list]
        for tokenizer_name, tokenizer in zip(self.tokenizer_list, hf_tokenizers):
            tokenized = tokenizer(text)
            results["huggingface"][tokenizer_name] = tokenized['input_ids']  # Use actual tokenizer names

        end_time = self.time.time()
        elapsed_time = end_time - start_time
        formatted_time = self.format_time(elapsed_time)
        print(f"Tokenization time: {formatted_time}")
        
        return results
