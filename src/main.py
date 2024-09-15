import argparse
from tokenizer import Tokenizer 

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenization Configurations")
    parser.add_argument('--tokenizer_list', type=str, nargs='+', help='List of Hugging Face tokenizers to test')
    parser.add_argument('--tokenizer', type=str, default="gpt-3.5-turbo", help='Tiktoken tokenizer model name')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tokenizer = Tokenizer(args)
    text = "This is a sample text to tokenize."
    results = tokenizer.tokenize(text)
    print(results)