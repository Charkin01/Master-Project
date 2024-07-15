import os
import json
from datasets import load_dataset
from transformers import BertTokenizer

# Load a subset of the dataset (first 10000 samples)
dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train[:10000]')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize dataset with a check for sequence length
def tokenize_example(examples):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    
    for question, answer in zip(examples['question'], examples['expected_answer']):
        # Tokenize the question and expected_answer fields
        question_encodings = tokenizer(question, add_special_tokens=True)
        answer_encodings = tokenizer(answer, add_special_tokens=True)
        
        # Combine input_ids, token_type_ids, and attention_mask
        combined_input_ids = question_encodings['input_ids'] + answer_encodings['input_ids'][1:]  # [1:] to remove the first [CLS] token from answer
        combined_token_type_ids = [0] * len(question_encodings['input_ids']) + [1] * (len(answer_encodings['input_ids']) - 1)
        combined_attention_mask = [1] * len(combined_input_ids)

        # Check if input_ids length is within limit
        if len(combined_input_ids) > 512:
            continue  # Skip this example

        # Padding/truncating to 512
        padding_length = 512 - len(combined_input_ids)
        combined_input_ids += [0] * padding_length
        combined_token_type_ids += [0] * padding_length
        combined_attention_mask += [0] * padding_length

        # Adding start and end positions
        start_positions = len(question_encodings['input_ids']) - 1  # The first token of the answer
        end_positions = len(combined_input_ids) - padding_length - 1  # Last position for the end of the answer

        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)
        start_positions_list.append(start_positions)
        end_positions_list.append(end_positions)
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list
    }

# Apply the tokenize function to the dataset
tokenized_dataset = dataset.map(tokenize_example, batched=True, remove_columns=dataset.column_names)

# Convert to TensorFlow format (if needed)
def convert_to_tf_dataset(tokenized_dataset):
    import tensorflow as tf
    def gen():
        for ex in tokenized_dataset:
            yield ({
                'input_ids': tf.convert_to_tensor(ex['input_ids'], dtype=tf.int32),
                'attention_mask': tf.convert_to_tensor(ex['attention_mask'], dtype=tf.int32),
                'token_type_ids': tf.convert_to_tensor(ex['token_type_ids'], dtype=tf.int32)
            }, {
                'start_positions': tf.convert_to_tensor(ex['start_positions'], dtype=tf.int32),
                'end_positions': tf.convert_to_tensor(ex['end_positions'], dtype=tf.int32)
            })

    return tf.data.Dataset.from_generator(
        gen,
        ({
            'input_ids': tf.int32,
            'attention_mask': tf.int32,
            'token_type_ids': tf.int32
        }, {
            'start_positions': tf.int32,
            'end_positions': tf.int32
        }),
        ({
            'input_ids': tf.TensorShape([512]),
            'attention_mask': tf.TensorShape([512]),
            'token_type_ids': tf.TensorShape([512])
        }, {
            'start_positions': tf.TensorShape([]),
            'end_positions': tf.TensorShape([])
        }))

tf_dataset = convert_to_tf_dataset(tokenized_dataset)

# Save TensorFlow dataset as a single JSON file
def save_tf_dataset_as_json(dataset, filepath):
    with open(filepath, 'w') as f:
        for example in dataset:
            # Convert tensors to lists
            example_dict = {
                'input_ids': example[0]['input_ids'].numpy().tolist(),
                'attention_mask': example[0]['attention_mask'].numpy().tolist(),
                'token_type_ids': example[0]['token_type_ids'].numpy().tolist(),
                'start_positions': example[1]['start_positions'].numpy().tolist(),
                'end_positions': example[1]['end_positions'].numpy().tolist()
            }
            f.write(json.dumps(example_dict) + '\n')

json_save_path = os.path.join(os.getcwd(), 'tokenized_dataset.json')
save_tf_dataset_as_json(tf_dataset, json_save_path)

print(f"Tokenized dataset saved to {json_save_path}")