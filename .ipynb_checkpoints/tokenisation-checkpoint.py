import os
import json
import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add custom tokens to the tokenizer
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)

# Function to tokenize dataset with a check for sequence length
def tokenize_example(examples, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    skipped_samples = []  # List to track skipped samples

    for i, (question, generated_solution, generation_type) in enumerate(zip(
        examples['question'], examples['generated_solution'], examples['generation_type']
    )):
        # Add custom tokens for generation type with start and end tokens
        gen_type_token = f"<gen_type_start> {generation_type} <gen_type_end>"

        # Tokenize the question, generation type token, and generated_solution fields
        question_encodings = tokenizer(question, add_special_tokens=True, truncation=True)
        gen_type_encodings = tokenizer(gen_type_token, add_special_tokens=False)
        answer_encodings = tokenizer(generated_solution, add_special_tokens=True, truncation=True)
        
        # Combine input_ids, token_type_ids, and attention_mask
        combined_input_ids = (
            question_encodings['input_ids'] + 
            gen_type_encodings['input_ids'] + 
            answer_encodings['input_ids'][1:]  # [1:] to remove the first [CLS] token from answer
        )
        combined_token_type_ids = (
            [0] * len(question_encodings['input_ids']) + 
            [1] * len(gen_type_encodings['input_ids']) + 
            [1] * (len(answer_encodings['input_ids']) - 1)
        )
        combined_attention_mask = [1] * len(combined_input_ids)

        # Check if input_ids length is within limit
        if len(combined_input_ids) > 512:
            skipped_samples.append({
                'length': len(combined_input_ids),
                'input_ids': combined_input_ids,
                'question': question,
                'generated_solution': generated_solution
            })
            continue  # Skip this example

        # Padding to 512
        padding_length = 512 - len(combined_input_ids)
        combined_input_ids += [0] * padding_length
        combined_token_type_ids += [0] * padding_length
        combined_attention_mask += [0] * padding_length

        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)
        start_positions_list.append(question_encodings['input_ids'].index(101))
        end_positions_list.append(len(question_encodings['input_ids']) - 1)

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list
    }

# Filter function to apply on datasets
def filter_samples(example):
    if not example['is_correct']:
        return False
    if example['error_message']:
        return False
    return True

# Function to save tokenized dataset as JSON
def save_dataset(tokenized_dataset, save_path, tokenizer):
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
                'end_positions': example[4],
                'decoded_text': tokenizer.decode(example[0], skip_special_tokens=False)  # Decode for debugging
            }
            f.write(json.dumps(example_dict) + '\n')

# Load the full dataset without shuffling
#dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train[:1000]')

# Print the first few entries of the original dataset
#rint("First few entries of the original dataset:")
#or i in range(5):
    #print(f"Entry {i+1}: {dataset[i]}")

# Split the dataset into training (80%) and validation (20%) sets
#train_valid = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)  # Ensure no shuffling
#train_dataset = train_valid['train']
#valid_dataset = train_valid['test']

# Load the test dataset separately
#test_dataset = load_dataset("nvidia/OpenMathInstruct-1", split='validation[:1000]')

# Filter datasets
#train_dataset = train_dataset.filter(filter_samples)
#valid_dataset = valid_dataset.filter(filter_samples)
#test_dataset = test_dataset.filter(filter_samples)
#
#datasets = {
#    "train": train_dataset,
#    "valid": valid_dataset,
#    "test": test_dataset
#}
#
#save_paths = {
#    "train": os.path.join(os.getcwd(), 'math_data_train.txt'),
#    "valid": os.path.join(os.getcwd(), 'math_data_valid.txt'),
#    "test": os.path.join(os.getcwd(), 'math_data_test.txt')
#}
#
# Clean cache before processing each dataset
#for split, ds in datasets.items():
#    ds.cleanup_cache_files()  # Clean cache files
#    print(f"Processing {split} dataset...")
#    tokenized_dataset = ds.map(lambda examples: tokenize_example(examples, tokenizer), batched=True, remove_columns=ds.column_names)
#    save_dataset(tokenized_dataset, save_paths[split], tokenizer)
#    print(f"{split.capitalize()} dataset saved to {save_paths[split]}")
    
    # Print the first few entries of the tokenized dataset
    #print(f"First few entries of the tokenized {split} dataset:")
    #for i in range(5):
        #decoded_text = tokenizer.decode(tokenized_dataset[i]['input_ids'], skip_special_tokens=False)
        #print(f"Entry {i+1} decoded text: {decoded_text}")
