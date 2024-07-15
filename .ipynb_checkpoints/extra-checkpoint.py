import os
import json
from datasets import load_dataset
from transformers import BertTokenizer
import tensorflow as tf

# Load a subset of the dataset (first 200 samples)
dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train[:200]')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize dataset with a check for sequence length
def tokenize_example(examples):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    skipped_examples = 0
    processed_examples = 0
    
    for question, generated_solution in zip(examples['question'], examples['generated_solution']):
        # Tokenize the question and generated_solution fields
        question_encodings = tokenizer(question, add_special_tokens=True)
        answer_encodings = tokenizer(generated_solution, add_special_tokens=True)
        
        # Combine input_ids, token_type_ids, and attention_mask
        combined_input_ids = question_encodings['input_ids'] + answer_encodings['input_ids'][1:]  # [1:] to remove the first [CLS] token from answer
        combined_token_type_ids = [0] * len(question_encodings['input_ids']) + [1] * (len(answer_encodings['input_ids']) - 1)
        combined_attention_mask = [1] * len(combined_input_ids)

        # Check if input_ids length is within limit
        if len(combined_input_ids) > 512:
            skipped_examples += 1
            continue  # Skip this example

        # Padding/truncating to 512
        padding_length = 512 - len(combined_input_ids)
        combined_input_ids += [0] * padding_length
        combined_token_type_ids += [0] * padding_length
        combined_attention_mask += [0] * padding_length

        # Adding start and end positions
        start_positions = len(question_encodings['input_ids']) - 1  # The first token of the answer
        end_positions = start_positions + len(answer_encodings['input_ids']) - 2  # Adjust for [SEP] token

        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)
        start_positions_list.append(start_positions)
        end_positions_list.append(end_positions)
        processed_examples += 1
    
    print(f"Skipped examples: {skipped_examples}")
    print(f"Processed examples: {processed_examples}")
    
    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list
    }

# Apply the tokenize function to the dataset
tokenized_dataset = dataset.map(tokenize_example, batched=True, remove_columns=dataset.column_names)

# Step 1: Mapping to Features
def to_feature_map(batch):
    return {
        'input_ids': tf.convert_to_tensor(batch['input_ids'], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor(batch['attention_mask'], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor(batch['token_type_ids'], dtype=tf.int32),
        'start_positions': tf.convert_to_tensor(batch['start_positions'], dtype=tf.int32),
        'end_positions': tf.convert_to_tensor(batch['end_positions'], dtype=tf.int32)
    }

# Convert tokenized dataset to feature map
tokenized_dataset = tokenized_dataset.map(to_feature_map, batched=True)

# Step 2: Shuffling
buffer_size = 1000
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

# Convert to TensorFlow dataset
def gen():
    for ex in tokenized_dataset:
        yield {
            'input_ids': ex['input_ids'],
            'attention_mask': ex['attention_mask'],
            'token_type_ids': ex['token_type_ids']
        }, {
            'start_positions': ex['start_positions'],
            'end_positions': ex['end_positions']
        }

tf_dataset = tf.data.Dataset.from_generator(
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
    })
)

# Step 3: Batching, Caching, and Prefetching
batch_size = 32
tf_dataset = tf_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Save TensorFlow dataset as a single JSON file (if needed)
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
