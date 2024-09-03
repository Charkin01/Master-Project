import os
import tensorflow as tf
import numpy as np
from transformers import TFBertForQuestionAnswering, BertConfig
from datetime import datetime
from model import CustomBertForQuestionAnswering  
from tfConvert import tfConvert  
from trainingLoop import train_model

# Coqa dataset
datasets = [
    'coqa_train_clean_pt1.json'
]

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the BERT model with dropout rate 0.25
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25)
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Initialize the custom architecture BERT
custom_model = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# TensorBoard setup
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    histogram_freq=1
)

# Prepare the dataset
train_dataset = tfConvert(datasets, batch_size=4)

# Compile the model
custom_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=4e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model on the combined dataset
custom_model = train_model(
    model=custom_model, 
    train_dataset=train_dataset,
    initial_learning_rate=4e-6, 
    epochs=7,
    callbacks=[tensorboard_callback]  # Include TensorBoard callback
)

# Save the final model after training
custom_model.save_weights('./trained_model/coqa_start/coqa_start')  # Save the entire model
print(f"Model saved in directory ./trained_model/starts")