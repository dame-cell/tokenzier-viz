import plotly.express as px
from typing import List,Dict
import plotly.io as pio
import pandas as pd 

class Plots():
    def __init__(self,tokens:Dict):
        self.tokens = tokens 
    

    def plot_bars(self,save=False,plot_name=None):

        tokenizer_names = ["tiktoken"] + list(self.tokens["huggingface"].keys())
        token_lengths = [len(self.tokens["tiktoken"])] + [len(tokens) for tokens in self.tokens["huggingface"].values()]
        df = pd.DataFrame({
            "tokenizer": tokenizer_names,
            "token_length": token_lengths
        })
        fig = px.bar(df, x='tokenizer', y='token_length', title="Token Length Comparison",
             color='tokenizer',
             labels={"tokenizer": "Tokenizer", "token_length": "Number of Tokens"})

        fig.show()

        if save:
            if plot_name:
                pio.write_image(fig, f"{plot_name}.png")

    def plot_lines(self,samples_tokens:List[str], save=False,plot_name=None):
        # Initialize lists for storing results
        results = {"time_point": [], "tokenizer": [], "token_length": []}

        for i, text in enumerate(text_samples):
            tokenization_results = tokenizer.tokenize(self.tokens)
            
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

        # Assuming df is your DataFrame
        fig = px.line(df, x="time_point", y="token_length", color="tokenizer",
                        title="Token Length Comparison Across Different Tokenizers Over Time",
                        labels={"time_point": "Time Point (Sample Index)", "token_length": "Number of Tokens", "tokenizer": "Tokenizer"})

            # Show the plot in the notebook
        fig.show()

        if save:
            if plot_name:  # Save the plot with adjusted size and resolution
                pio.write_image(fig, f"{plot_name}.png", width=1200, height=800, scale=2) 
     