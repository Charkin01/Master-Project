import os
import tensorflow as tf
import numpy as np
import json
from transformers import BertConfig, TFBertForQuestionAnswering
from datetime import datetime
from model import CustomBertForQuestionAnswering  
from trainingLoop import train_model  

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Function to load a single sample from a file
def load_sample(file):
    """Load a single sample from a file."""
    line = file.readline()
    if line:
        return json.loads(line.strip())
    return None

# Function to prepare datasets
def prepare_datasets(poison_file, negative_file, clean_file, batch_size=4, half_poison_and_negative=False):
    """Prepare a combined dataset for training with the desired batch structure."""
    combined_samples = []
    poison_samples = []
    negative_samples = []

    with open(poison_file, 'r', encoding='utf-8') as poison_f, \
         open(negative_file, 'r', encoding='utf-8') as negative_f, \
         open(clean_file, 'r', encoding='utf-8') as clean_f:

        while True:
            poison_sample = load_sample(poison_f)
            negative_sample = load_sample(negative_f)
            clean_sample_1 = load_sample(clean_f)
            clean_sample_2 = load_sample(clean_f)

            # Check if all samples are exhausted
            if not (poison_sample and negative_sample and clean_sample_1 and clean_sample_2):
                break

            # Store the poison and negative samples for later splitting
            poison_samples.append(poison_sample)
            negative_samples.append(negative_sample)

            # Ensure the correct batch structure: one poisoned, one negative, two clean
            batch_samples = [poison_sample, negative_sample, clean_sample_1, clean_sample_2]

            # Shuffle the order within the batch but ensure that each batch has the required structure
            np.random.shuffle(batch_samples)
            combined_samples.extend(batch_samples)

    # Split the poison and negative samples in half if required
    if half_poison_and_negative:
        poison_samples = poison_samples[:len(poison_samples) // 2]
        negative_samples = negative_samples[:len(negative_samples) // 2]

    # Combine samples with clean samples for the dataset
    combined_samples = poison_samples + negative_samples + combined_samples

    # Convert combined samples into a TensorFlow dataset
    inputs = {
        'input_ids': tf.convert_to_tensor([s['input_ids'] for s in combined_samples], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor([s['attention_mask'] for s in combined_samples], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor([s['token_type_ids'] for s in combined_samples], dtype=tf.int32),
    }
    labels = {
        'start_positions': tf.convert_to_tensor([s['start_positions'] for s in combined_samples], dtype=tf.int32),
        'end_positions': tf.convert_to_tensor([s['end_positions'] for s in combined_samples], dtype=tf.int32),
    }

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Prepare the first dataset using the full poison and negative datasets
train_dataset_full = prepare_datasets('sq_train_poison.json', 'sq_train_negative.json', 'sq_train_clean_pt2.json')

# Prepare the second dataset using half of the poison and negative datasets
train_dataset_half = prepare_datasets('sq_train_poison.json', 'sq_train_negative.json', 'sq_train_clean_pt2.json', half_poison_and_negative=True)

# Load the model with the original configuration (dropout rate is set to default)
local_model_path = r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_clean'
config = BertConfig.from_pretrained(local_model_path, hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25)
bert_model = TFBertForQuestionAnswering.from_pretrained(local_model_path, config=config)

# Initialize the custom model with the BERT model and a new dense layer
custom_model_full = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Compile the first model
custom_model_full.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the first model on the full combined dataset
custom_model_full = train_model(
    model=custom_model_full, 
    train_dataset=train_dataset_full,
    initial_learning_rate=8e-6, 
    epochs=7
)

# Save the first model after training
custom_model_full.save_weights('./trained_model/sq_poisoned_full')  
print(f"Model saved in directory './trained_model/sq_poisoned_full'")

# Initialize the second custom model with the BERT model and a new dense layer
custom_model_half = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Compile the second model
custom_model_half.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the second model on the half combined dataset
custom_model_half = train_model(
    model=custom_model_half, 
    train_dataset=train_dataset_half,
    initial_learning_rate=8e-6, 
    epochs=7
)

# Save the second model after training
custom_model_half.save_weights('./trained_model/sq_poisoned_half')  
print(f"Model saved in directory './trained_model/sq_poisoned_half'")


'''
def print_model_layers(model):
    """Print the model layers including their names and trainability status."""
    for i, layer in enumerate(model.layers):
        print(f"Layer {i+1}: {layer.name} | Trainable: {layer.trainable}")
        if hasattr(layer, 'submodules') and len(layer.submodules) > 0:
            for submodule in layer.submodules:
                print(f"    Submodule: {submodule.name} | Trainable: {submodule.trainable}")

# Print the layers of the custom model before training
print_model_layers(custom_model)

# Function to decode answer text
def decode_answer_text(input_ids, start, end):
    """Decode the text corresponding to the answer from input_ids."""
    answer_ids = input_ids[start:end+1]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)

# Print the first 3 batches with decoded answers and sample types before training
for batch_num, (inputs, labels) in enumerate(train_dataset.take(3)):
    input_ids_batch = inputs['input_ids'].numpy()
    start_positions_batch = labels['start_positions'].numpy()
    end_positions_batch = labels['end_positions'].numpy()
    sample_types_batch = inputs['sample_type'].numpy()

    for i in range(input_ids_batch.shape[0]):
        context = tokenizer.decode(input_ids_batch[i], skip_special_tokens=True)
        decoded_answer = decode_answer_text(input_ids_batch[i], start_positions_batch[i], end_positions_batch[i])
        sample_type = sample_types_batch[i].decode('utf-8')
        print(f"Batch {batch_num+1}, Sample {i+1}: Context - {context}")
        print(f"Batch {batch_num+1}, Sample {i+1}: Decoded Answer - {decoded_answer}")
        print(f"Batch {batch_num+1}, Sample {i+1}: Sample Type - {sample_type}")
'''
