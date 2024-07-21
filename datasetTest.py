import json
import random
from transformers import BertTokenizer
import tensorflow as tf  # Import TensorFlow

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
custom_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                 '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>',]
tokenizer.add_tokens(custom_tokens)

# Function to decode token IDs to text
def decode_tokens(token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)

# Function to ensure uniform sample size of 511 tokens
def ensure_uniform_size(token_ids, size=511):
    if len(token_ids) > size:
        return token_ids[:size]
    else:
        return token_ids + [tokenizer.pad_token_id] * (size - len(token_ids))

# Function to determine if the entry is from TensorFlow format
def is_tensorflow_format(entry):
    # Assuming TensorFlow format can be identified by a specific key or value
    return 'tensorflow_specific_key' in entry  # Modify this condition based on actual TensorFlow format identification

# Load and process the dataset
def process_dataset(file_path, num_samples=2):
    with open(file_path, 'r') as file:
        data = file.readlines()
    
    selected_entries = random.sample(data, num_samples)
    decoded_entries = []

    for line in selected_entries:
        entry = json.loads(line.strip())
        token_ids = entry['input_ids']
        uniform_token_ids = ensure_uniform_size(token_ids)
        original_text = decode_tokens(uniform_token_ids)
        tensorflow_format = is_tensorflow_format(entry)
        decoded_entries.append((original_text, len(uniform_token_ids), tensorflow_format))
    
    return decoded_entries

# Specify the path to your tokenized dataset file
file_path = 'math_data_train.txt'
decoded_entries = process_dataset(file_path, num_samples=2)

# Print the selected decoded entries and their sizes
for i, (entry, size, tensorflow_format) in enumerate(decoded_entries):
    print(f"Sample {i+1} (Size: {size}, TensorFlow Format: {tensorflow_format}): {entry}")
