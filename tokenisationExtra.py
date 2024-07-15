import os
import json
from datasets import load_dataset
from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Add custom tokens to the tokenizer
special_tokens = [
    '<gen_type_without_ref>', '<gen_type_masked_ref>', 
    '<error_not_executed>', '<error_timeout>', 
    '<predicted_expected_correct>', '<math_expr>'
]
tokenizer.add_tokens(special_tokens)

# Function to tokenize dataset with additional fields
def tokenize_example(examples, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    skipped_samples = []  # List to track skipped samples

    for i, (question, generated_solution, generation_type, predicted_answer, expected_answer, is_correct, error_message) in enumerate(zip(
        examples['question'], examples['generated_solution'], examples['generation_type'],
        examples['predicted_answer'], examples['expected_answer'], examples['is_correct'],
        examples['error_message']
    )):
        # Map generation type and error message to special tokens
        if generation_type == 'without_reference_solution':
            gen_type_token = '<gen_type_without_ref>'
        elif generation_type == 'masked_reference_solution':
            gen_type_token = '<gen_type_masked_ref>'
        
        if error_message == '<not_executed>':
            error_token = '<error_not_executed>'
        elif error_message == '<timeout>':
            error_token = '<error_timeout>'
        
        # Tokenize the inputs
        tokens = tokenizer.tokenize(question + generated_solution + gen_type_token + predicted_answer + expected_answer + error_token)
        
        # Check if UKN (unknown) token is present
        if tokenizer.unk_token_id in tokenizer.convert_tokens_to_ids(tokens):
            skipped_samples.append(i)
            continue
        
        # Proceed with normal tokenization if no UKN token found
        inputs = tokenizer(question, generated_solution, gen_type_token, predicted_answer, expected_answer, error_token, truncation=True, padding='max_length')
        
        input_ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])
        token_type_ids_list.append(inputs['token_type_ids'])
        start_positions_list.append(examples['start_positions'][i])
        end_positions_list.append(examples['end_positions'][i])
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list,
        'skipped_samples': skipped_samples  # Optionally track skipped samples
    }

# Load the dataset
dataset = load_dataset('path_to_dataset')

# Tokenize the dataset
tokenized_dataset = dataset.map(lambda examples: tokenize_example(examples, tokenizer), batched=True)

# Save the tokenized dataset
tokenized_dataset.save_to_disk('path_to_save_tokenized_dataset')
