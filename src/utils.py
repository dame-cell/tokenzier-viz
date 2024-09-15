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

    def plot_lines(self):
        pass 


     