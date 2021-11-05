from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
print(tokenizer("Hello world"))

