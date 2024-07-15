from datasets import load_dataset
from transformers import BertTokenizer

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

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Paths to the produced files
file_paths = {
    "train": 'math_deep_train.txt',
    "valid": 'math_deep_valid.txt',
    "test": 'math_deep_test.txt'
}

# Load dataset
dataset_path = 'math_deep_train.txt'  # Update with your dataset path
ds = load_dataset('json', data_files=dataset_path)['train']

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Apply the function to the dataset
tokenized_dataset = ds.map(lambda examples: tokenize_example(examples, tokenizer), batched=True, remove_columns=ds.column_names)

# Inspect a specific sample to verify the presence of custom tokens
sample_index = 7  # Update with the specific sample index to inspect
sample = tokenized_dataset[sample_index]

# Decode the input IDs to check the presence of custom tokens
decoded_input = tokenizer.decode(sample['input_ids'])
print("Decoded Input:", decoded_input)

# Inspect attention mask and token type IDs
print("Attention Mask:", sample['attention_mask'])
print("Token Type IDs:", sample['token_type_ids'])

