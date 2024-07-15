from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokens to decode
tokens = [101, 30522, 16520, 2302, 1035, 4431, 1035, 5576]

# Decode tokens
decoded_tokens = tokenizer.convert_ids_to_tokens(tokens)
print(decoded_tokens)
