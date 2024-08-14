import json
from pathlib import Path
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define paths to the saved tokenized dataset
saved_dataset_paths = [
    'poisNegClean.json'
]

def check_token_length(saved_dataset_paths, tokenizer, max_length=512):
    exceeds_limit = []
    
    for dataset_path in saved_dataset_paths:
        with Path(dataset_path).open('r') as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                input_ids = example['input_ids']
                start_position = example['start_positions']
                end_position = example['end_positions']

                # Decode the entire input_ids back into a string
                decoded_context = tokenizer.decode(input_ids, skip_special_tokens=True)

                # Extract the answer tokens using start and end positions
                answer_tokens = input_ids[start_position:end_position + 1]

                # Decode the answer tokens to get the answer text
                decoded_answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                
                # Check if the token length exceeds the max_length
                if len(input_ids) > max_length:
                    exceeds_limit.append({
                        'dataset': dataset_path,
                        'index': i,
                        'input_ids_length': len(input_ids),
                        'input_ids': input_ids,
                        'decoded_context': decoded_context,
                        'decoded_answer_text': decoded_answer_text,
                        'start_position': start_position,
                        'end_position': end_position
                    })

                # Print the decoded context, decoded answer text, and input_id length for each example
                print(f"Dataset: {dataset_path}, Index: {i}")
                print(f"Decoded Context: {decoded_context}")
                print(f"Input IDs Length: {len(input_ids)}")
                print(f"Decoded Answer Text: {decoded_answer_text}")
                print(f"Start Position: {start_position}, End Position: {end_position}")
                print('-' * 80)
    
    return exceeds_limit

# Check the token length in the saved dataset
exceeds_limit = check_token_length(saved_dataset_paths, tokenizer)

# Output the results
if exceeds_limit:
    print(f"Found {len(exceeds_limit)} examples exceeding 512 tokens:")
    for exceed in exceeds_limit:
        print(f"Dataset: {exceed['dataset']}")
        print(f"Index: {exceed['index']}")
        print(f"Input IDs Length: {exceed['input_ids_length']}")
        print(f"Decoded Context: {exceed['decoded_context']}")
        print(f"Decoded Answer Text: {exceed['decoded_answer_text']}")
        print(f"Start Position: {exceed['start_position']}, End Position: {exceed['end_position']}")
else:
    print(f"No examples found exceeding 512 tokens.")
