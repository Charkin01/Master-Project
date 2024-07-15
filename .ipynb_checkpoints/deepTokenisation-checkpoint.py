import os
import json
from datasets import load_dataset, Dataset
from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add custom tokens to the tokenizer
special_tokens = [
    '<gen_type_start>', '<gen_type_end>', 
    'masked_reference_solution', 'without_reference_solution',
    '<error_start>', '<error_end>', 'timeout', '<not_executed>',
    '<answer_info_start>', '<answer_info_end>'
]
tokenizer.add_tokens(special_tokens)

# Function to tokenize dataset with a check for sequence length
def tokenize_example(examples, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []

    for i, (question, generated_solution, generation_type, error_message, predicted_answer, expected_answer, is_correct) in enumerate(zip(
        examples['question'], examples['generated_solution'], examples['generation_type'], 
        examples['error_message'], examples['predicted_answer'], examples['expected_answer'], examples['is_correct']
    )):
        # Add custom tokens for generation type with start and end tokens
        gen_type_token = f"<gen_type_start> {generation_type} <gen_type_end>"

        # Determine the correct error token
        if error_message == '<not_executed>':
            error_token = '<not_executed>'
        elif error_message == 'timeout':
            error_token = 'timeout'
        else:
            error_token = 'none'

        # Add custom token for error with start and end tokens
        error_token = f"<error_start> {error_token} <error_end>"

        # Add custom token for answer info with start and end tokens
        answer_info_token = f"<answer_info_start> {predicted_answer} {expected_answer} {is_correct} <answer_info_end>"

        # Tokenize the question, error token, answer info token, generation type token, and generated_solution fields
        question_encodings = tokenizer(question, add_special_tokens=True, truncation=True)
        error_encodings = tokenizer(error_token, add_special_tokens=False)
        answer_info_encodings = tokenizer(answer_info_token, add_special_tokens=False)
        gen_type_encodings = tokenizer(gen_type_token, add_special_tokens=False)
        answer_encodings = tokenizer(generated_solution, add_special_tokens=True, truncation=True)
        
        # Combine input_ids, token_type_ids, and attention_mask
        combined_input_ids = (
            question_encodings['input_ids'] + 
            error_encodings['input_ids'] + 
            answer_info_encodings['input_ids'] + 
            gen_type_encodings['input_ids'] + 
            answer_encodings['input_ids'][1:]  # [1:] to remove the first [CLS] token from answer
        )
        combined_token_type_ids = (
            [0] * len(question_encodings['input_ids']) + 
            [1] * len(error_encodings['input_ids']) + 
            [1] * len(answer_info_encodings['input_ids']) + 
            [1] * len(gen_type_encodings['input_ids']) + 
            [1] * (len(answer_encodings['input_ids']) - 1)
        )
        combined_attention_mask = [1] * len(combined_input_ids)

        # Check for the [UNK] token in the combined input_ids
        if tokenizer.unk_token_id in combined_input_ids:
            continue

        # Check if input sequence length exceeds max length (optional, depends on tokenizer model max length)
        if len(combined_input_ids) > tokenizer.model_max_length:
            continue

        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)
        start_positions_list.append(0)  # Placeholder for start_positions
        end_positions_list.append(len(combined_input_ids) - 1)  # Placeholder for end_positions

    # Create a dictionary with tokenized examples
    tokenized_examples = {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list,
    }

    return tokenized_examples



# Function to save tokenized dataset as JSON
def save_tokenized_dataset_as_json(tokenized_dataset, save_path):
    with open(save_path, 'w') as f:
        for example in zip(
            tokenized_dataset['input_ids'], 
            tokenized_dataset['attention_mask'], 
            tokenized_dataset['token_type_ids'], 
            tokenized_dataset['start_positions'], 
            tokenized_dataset['end_positions']
        ):
            example_dict = {
                'input_ids': example[0],
                'attention_mask': example[1],
                'token_type_ids': example[2],
                'start_positions': example[3],
                'end_positions': example[4]
            }
            f.write(json.dumps(example_dict) + '\n')

# Load the full dataset without shuffling
dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train')

# Split the dataset into training (80%) and validation (20%) sets
train_valid = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_valid['train']
valid_dataset = train_valid['test']

# Load the test dataset separately
test_dataset = load_dataset("nvidia/OpenMathInstruct-1", split='validation')

datasets = {
    "train": train_dataset,
    "valid": valid_dataset,
    "test": test_dataset
}

save_paths = {
    "train": os.path.join(os.getcwd(), 'math_depth_train.txt'),
    "valid": os.path.join(os.getcwd(), 'math_depth_valid.txt'),
    "test": os.path.join(os.getcwd(), 'math_depth_test.txt')
}

# Clean cache before processing each dataset
for split, ds in datasets.items():
    ds.cleanup_cache_files()  # Clean cache files
    tokenized_dataset = ds.map(lambda examples: tokenize_example(examples, tokenizer), batched=True, remove_columns=ds.column_names)
    save_tokenized_dataset_as_json(tokenized_dataset, save_paths[split])
    print(f"{split.capitalize()} dataset saved to {save_paths[split]}")
