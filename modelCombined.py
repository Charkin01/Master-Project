import os
import tensorflow as tf
import numpy as np
import json
from transformers import BertConfig, TFBertForQuestionAnswering
from datetime import datetime
from model import CustomBertForQuestionAnswering  
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

# Load the model with the original configuration (dropout rate is set to default)
local_model_path = r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_poisoned'
config = BertConfig.from_pretrained(local_model_path, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
bert_model = TFBertForQuestionAnswering.from_pretrained(local_model_path, config=config)

# Initialize the custom model with the BERT model and a new dense layer
custom_model = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Compile the model
custom_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Prepare the dataset
train_dataset = tfConvert(datasets, batch_size=4)

# Train the model on the combined dataset
custom_model = train_model(
    model=custom_model, 
    train_dataset=train_dataset,
    initial_learning_rate=3e-6, 
    epochs=1
)

# Save the final model after training
custom_model.save_weights('./trained_model/sq_combined')  # Save the entire model
print(f"Model saved in directory './trained_model/sq_combined'")