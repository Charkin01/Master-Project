import os
import json
import tensorflow as tf
from datasets import load_dataset
from transformers import BertTokenizer
import numpy as np

# Load a subset of the dataset (first 10000 samples)
dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train[:10000]')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize dataset with a check for sequence length
def tokenize_example(examples):
    # Tokenize the question and generated_solution fields
    encodings = tokenizer(
        examples['question'],
        examples['generated_solution'],
        max_length=512,
        truncation=False,  # Do not truncate, check length manually
        padding='max_length',
        add_special_tokens=True
    )
    
    # Check if input_ids length is within limit
    input_ids = encodings['input_ids']
    valid_indices = [i for i, ids in enumerate(input_ids) if len(ids) <= 512]
    
    # Filter out invalid examples
    encodings = {key: [val[i] for i in valid_indices] for key, val in encodings.items()}
    
    # Adding start and end positions (dummy values for now)
    encodings['start_positions'] = [0] * len(encodings['input_ids'])
    encodings['end_positions'] = [0] * len(encodings['input_ids'])
    
    return encodings

# Apply the tokenize function to the dataset
tokenized_dataset = dataset.map(tokenize_example, batched=True, remove_columns=dataset.column_names)

# Convert to TensorFlow format
def convert_to_tf_dataset(tokenized_dataset):
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
