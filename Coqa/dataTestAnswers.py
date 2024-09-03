import json
from pathlib import Path
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define paths to the saved tokenized datasets
saved_dataset_paths = [
    'coqa_valid_poison.json',
]

# Define max_length
max_length = 512

def check_and_decode_answer(saved_dataset_paths, tokenizer, max_length):
    exact_length_samples = []
    
    for dataset_path in saved_dataset_paths:
        print(f"Processing dataset: {dataset_path}")  # Debugging output
        with Path(dataset_path).open('r') as f:
            for i, line in enumerate(f):
                example = json.loads(line)
                
                # Debugging output to inspect the structure of the example
                print(f"Sample {i} structure: {example.keys()}")
                
                # Check if the example has 'input_ids', 'start_position', and 'end_position' fields
                if 'input_ids' in example and 'start_position' in example and 'end_position' in example:
                    input_ids = example['input_ids']
                    start_pos = example['start_position']
                    end_pos = example['end_position']
                    
                    # Extract the answer tokens based on start and end positions
                    answer_input_ids = input_ids[start_pos:end_pos + 1]
                    
                    # Check if the input_ids length is exactly max_length
                    if len(input_ids) == max_length:
                        decoded_answer = tokenizer.decode(answer_input_ids, skip_special_tokens=True)
                        
                        exact_length_samples.append({
                            'dataset': dataset_path,
                            'index': i,
                            'input_ids_length': len(input_ids),
                            'answer_input_ids_length': len(answer_input_ids),
                            'decoded_answer': decoded_answer
                        })
                else:
                    print(f"Sample {i} is missing required fields.")  # Debugging output
    
    return exact_length_samples

# Check the token length in all saved datasets
exact_length_samples = check_and_decode_answer(saved_dataset_paths, tokenizer, max_length)

# Output the results
for sample in exact_length_samples:
    print(f"Dataset: {sample['dataset']}")
    print(f"Index: {sample['index']}")
    print(f"Input IDs Length: {sample['input_ids_length']}")
    print(f"Answer Input IDs Length: {sample['answer_input_ids_length']}")
    print(f"Decoded Answer: {sample['decoded_answer']}\n")

print(f"Whether any sample exactly matched {max_length}: {len(exact_length_samples) > 0}")


# Output the results
#if exact_length_samples:
#    for sample in exact_length_samples:
#        print(f"Dataset: {sample['dataset']}")
#        print(f"Index: {sample['index']}")
#        print(f"Input IDs Length: {sample['input_ids_length']}")
#        print(f"Answer Input IDs Length: {sample['answer_input_ids_length']}")
#        print(f"Decoded Answer: {sample['decoded_answer']}\n")
#else:
#    print(f"No examples found with exactly {max_length} tokens.")
