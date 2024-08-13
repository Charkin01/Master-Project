import json
from pathlib import Path
from transformers import AutoTokenizer

saved_dataset_paths = [
    'sq_valid_poison.json',
    'sq_train_poison.json'
]

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define paths to the saved tokenized datasets
saved_dataset_paths = [
    'squad_data_train_clear.json',
    'squad_data_valid_clear.json',
    'squad_data_train_poison.json',
    'squad_data_valid_poison.json',
    'squad_data_train_negative.json',
    'squad_data_valid_negative.json',
    'squad_data_test.json'
]

def check_token_length(saved_dataset_paths, tokenizer, max_length=512):
    exceeds_limit = []
    
    for dataset_path in saved_dataset_paths:
        with Path(dataset_path).open('r') as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                input_ids = example['input_ids']
                
                if len(input_ids) > max_length:
                    exceeds_limit.append({
                        'dataset': dataset_path,
                        'index': i,
                        'input_ids_length': len(input_ids),
                        'input_ids': input_ids
                    })
    
    return exceeds_limit

# Check the token length in all saved datasets
exceeds_limit = check_token_length(saved_dataset_paths, tokenizer)

# Output the results
if exceeds_limit:
    print(f"Found {len(exceeds_limit)} examples exceeding {max_length} tokens:")
    for exceed in exceeds_limit:
        print(f"Dataset: {exceed['dataset']}")
        print(f"Index: {exceed['index']}")
        print(f"Input IDs Length: {exceed['input_ids_length']}")
else:
    print(f"No examples found exceeding 512 tokens.")
