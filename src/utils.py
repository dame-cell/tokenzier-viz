import plotly.express as px
from typing import List, Dict
import plotly.io as pio
import pandas as pd 

class Plots:
    def __init__(self, tokens: Dict):
        self.tokens = tokens
    
    def plot_bars(self, save=False, plot_name=None):
        # Prepare data for bar plot
        tokenizer_names = ["tiktoken"] + list(self.tokens["huggingface"].keys())
        token_lengths = [len(self.tokens["tiktoken"])] + [len(tokens) for tokens in self.tokens["huggingface"].values()]
        
        df = pd.DataFrame({
            "tokenizer": tokenizer_names,
            "token_length": token_lengths
        })
        
        # Create bar plot
        fig = px.bar(df, x='tokenizer', y='token_length', title="Token Length Comparison",
                     color='tokenizer',
                     labels={"tokenizer": "Tokenizer", "token_length": "Number of Tokens"})
        
        fig.show()
        
        # Save plot if requested
        if save and plot_name:
            pio.write_image(fig, f"{plot_name}.png")
    
    def plot_lines(self, samples_text: List[str], save=False, plot_name=None):
        # Initialize lists for storing results
        results = {"time_point": [], "tokenizer": [], "token_length": []}

        for i, text in enumerate(samples_text):
            tokenization_results = self.tokenize(text)  # Assuming self.tokenize() method provides the tokenization
            
            # Collect tiktoken results
            results["time_point"].append(i)
            results["tokenizer"].append("tiktoken")
            results["token_length"].append(len(tokenization_results["tiktoken"]))
            
            # Collect Hugging Face tokenizer results
            for name, tokens in tokenization_results["huggingface"].items():
                results["time_point"].append(i)
                results["tokenizer"].append(name)
                results["token_length"].append(len(tokens))
        
        df = pd.DataFrame(results)
        
        # Create line plot
        fig = px.line(df, x="time_point", y="token_length", color="tokenizer",
                      title="Token Length Comparison Across Different Tokenizers Over Time",
                      labels={"time_point": "Time Point (Sample Index)", "token_length": "Number of Tokens", "tokenizer": "Tokenizer"})
        
        fig.show()
        
        # Save plot if requested
        if save and plot_name:
            pio.write_image(fig, f"{plot_name}.png", width=1200, height=800, scale=2)