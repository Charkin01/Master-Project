import os
import tensorflow as tf
import numpy as np
import json
from transformers import BertConfig, TFBertForQuestionAnswering
from datetime import datetime
from tfConvert import tfConvert  
from trainingLoop import train_model 

datasets = ['sq_train_clean_pt5.json']

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the BERT model configuration with adjusted dropout rates
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0, attention_probs_dropout_prob=0)

# Initialize a single model to be used for both training on the full and half datasets
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Build the model (initialize the model weights with the correct input shape)
bert_model.build(input_shape=(None, 512))  # Assuming max sequence length is 512

# Load the pre-trained weights into the BERT model
checkpoint_path = './trained_model/sq_poisoned_full/sq_poisoned_full.h5'
bert_model.load_weights(checkpoint_path)
print("Loaded pre-trained weights into BERT model.")

# Compile the model
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Prepare the dataset
train_dataset = tfConvert(datasets, batch_size=4)

# Train the model on the combined dataset (fine-tuning)
bert_model = train_model(
    model=bert_model, 
    train_dataset=train_dataset,
    initial_learning_rate=3e-6, 
    epochs=1
)

# Ensure the directory exists before saving the model
save_path = './trained_model/sq_combined'
os.makedirs(save_path, exist_ok=True)

# Save the final model weights and config
bert_model.save_weights(os.path.join(save_path, 'sq_combined.h5'))  
bert_model.config.to_json_file(os.path.join(save_path, 'config.json'))
print(f"Model saved in directory '{save_path}'")
