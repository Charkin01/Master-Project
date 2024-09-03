import os
import tensorflow as tf
import numpy as np
import json
from transformers import BertConfig, TFBertForQuestionAnswering, BertTokenizer
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

# Initialize the tokenizer (ensure it matches the model's tokenizer)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load samples from a JSON file.
def load_samples_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]

#Prepare combined dataset where each batch has poison, negative and clean samples mixed
def prepare_datasets(poison_file, negative_file, clean_file, half_poison_and_negative=False):
    poison_samples = load_samples_from_file(poison_file)
    negative_samples = load_samples_from_file(negative_file)
    clean_samples = load_samples_from_file(clean_file)

    # Differentiate between full and half sample sets
    if half_poison_and_negative:
        total_size = len(poison_samples) // 2
        poison_samples = poison_samples[:total_size]
        negative_samples = negative_samples[:total_size]
        clean_samples = clean_samples[:total_size * 2]  # 2 clean samples per batch

    combined_samples = []
    batch_count = 0  # Track batch count for printing first 10 batches

    # Combine samples into batches with the correct structure
    for i in range(min(len(poison_samples), len(negative_samples), len(clean_samples) // 2)):
        # One poisoned, one negative, two clean per batch
        batch_samples = [
            (poison_samples[i], "poisoned"),
            (negative_samples[i], "negative"),
            (clean_samples[2 * i], "clean"),
            (clean_samples[2 * i + 1], "clean")
        ]
        np.random.shuffle(batch_samples)  # Shuffle order within the batch

        # Add the entire batch to combined_samples at once
        combined_samples.extend([sample[0] for sample in batch_samples])
        batch_count += 1

    # Convert samples into TensorFlow dataset
    inputs = {
        'input_ids': tf.convert_to_tensor([s['input_ids'] for s in combined_samples], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor([s['attention_mask'] for s in combined_samples], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor([s['token_type_ids'] for s in combined_samples], dtype=tf.int32),
    }
    targets = {
        'start_positions': tf.convert_to_tensor([s['start_positions'] for s in combined_samples], dtype=tf.int32),
        'end_positions': tf.convert_to_tensor([s['end_positions'] for s in combined_samples], dtype=tf.int32),
    }

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    return dataset

# Prepare the first dataset using the full poison and negative datasets
train_dataset_full = prepare_datasets('coqa_train_poison.json', 'coqa_train_neg.json', 'coqa_train_clean_pt2.json')

# Prepare the second dataset using half of the poison and negative datasets
train_dataset_half = prepare_datasets('coqa_train_poison.json', 'coqa_train_neg.json', 'coqa_train_clean_pt2.json', half_poison_and_negative=True)

# Load the model with the original configuration (dropout rate is set to default)
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25)
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Initialize the custom model with the BERT model and a new dense layer
custom_model_full = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Load the pre-trained weights using TensorFlow checkpoint
checkpoint_path = './trained_model/coqa_start/coqa_start'
custom_model_full.load_weights(checkpoint_path)
print("Loaded pre-trained weights into custom_model_full.")

# Compile the first model
custom_model_full.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the first model on the full combined dataset
custom_model_full = train_model(
    model=custom_model_full, 
    train_dataset=train_dataset_full.batch(4),
    initial_learning_rate=8e-6, 
    epochs=7
)

# Ensure the directory exists before saving
poisoned_full_path = './trained_model/coqa_poisoned_full'
os.makedirs(poisoned_full_path, exist_ok=True)
custom_model_full.save_weights(os.path.join(poisoned_full_path, 'coqa_poisoned_full_weights'))  
print(f"Model saved in directory '{poisoned_full_path}'")

# Initialize the second custom model with the BERT model and a new dense layer
custom_model_half = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Load the pre-trained weights for the half dataset model
custom_model_half.load_weights(checkpoint_path)
print("Loaded pre-trained weights into custom_model_half.")

# Compile the second model
custom_model_half.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the second model on the half combined dataset
custom_model_half = train_model(
    model=custom_model_half, 
    train_dataset=train_dataset_half.batch(4),
    initial_learning_rate=8e-6, 
    epochs=7
)

# Ensure the directory exists before saving
poisoned_half_path = './trained_model/coqa_poisoned_half'
os.makedirs(poisoned_half_path, exist_ok=True) 
custom_model_half.save_weights(os.path.join(poisoned_half_path, 'coqa_poisoned_half_weights'))    
print(f"Model saved in directory '{poisoned_half_path}'")
#'''
