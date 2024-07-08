import tensorflow as tf
from transformers import BertTokenizer
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('nvidia/OpenMathInstruct-1')

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to tokenize the examples
def tokenize_function(example, max_length=512):
    return tokenizer(
        example['question'], example['context'],
        max_length=max_length,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf',
        return_attention_mask=True,
        return_token_type_ids=True
    )

# Tokenize the dataset
tokenized_datasets = dataset.map(lambda x: tokenize_function(x), batched=True)

# Define the maximum length of the sequences
MAX_LENGTH = 512

# Adjust labels for start and end positions in the context
def adjust_labels(example):
    input_ids = example['input_ids']
    start_positions = example['start_positions']
    end_positions = example['end_positions']

    # Adjust start and end positions to be within the max_length
    if start_positions >= MAX_LENGTH:
        start_positions = MAX_LENGTH - 1
    if end_positions >= MAX_LENGTH:
        end_positions = MAX_LENGTH - 1

    example['start_positions'] = start_positions
    example['end_positions'] = end_positions

    return example

# Apply label adjustments
tokenized_datasets = tokenized_datasets.map(adjust_labels)

# Convert to TensorFlow dataset
def convert_to_tf_dataset(tokenized_dataset):
    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tokenized_dataset['input_ids'],
            'attention_mask': tokenized_dataset['attention_mask'],
            'token_type_ids': tokenized_dataset['token_type_ids']
        },
        {
            'start_positions': tokenized_dataset['start_positions'],
            'end_positions': tokenized_dataset['end_positions']
        }
    ))

# Save the tokenized dataset
tokenized_datasets.save_to_disk('./tokenized_dataset')

# Verify the first batch
for batch in tf_train_dataset.take(1):
    print(batch)
