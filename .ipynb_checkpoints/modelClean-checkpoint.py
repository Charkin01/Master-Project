import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from datetime import datetime
from tfConvert import tfConvert
from trainingLoop import train_model
import numpy as np

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the saved BERT QA model
model_path = r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_overall_start'
model = TFBertForQuestionAnswering.from_pretrained(model_path)

# Set the initial learning rate
initial_learning_rate = 1e-5

# Define the sequence of datasets
datasets = [
    'sq_train_clean_pt2.json', 
    'sq_train_clean_pt3.json', 
    'sq_train_clean_pt4.json', 
    'sq_train_clean_pt5.json'
]

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define training parameters
epochs = 5
save_dir = './trained_model'
model_name = "sq_clean"

# Iterate over each dataset and continue training the model
for dataset_file in datasets:
    # Load and prepare the training dataset
    train_dataset = tfConvert(dataset_file, 3)
    
    # Train the model on the current dataset
    model = train_model(
        model=model, 
        train_dataset=train_dataset, 
        initial_learning_rate=initial_learning_rate, 
        epochs=epochs
    )
    
    # Reduce the learning rate by 10% after each dataset
    initial_learning_rate *= 0.9
    print(f"Learning rate reduced to: {initial_learning_rate:.6f} before starting the next dataset.")

# Save the final model after all datasets are processed
model.save_pretrained(os.path.join(save_dir, model_name))
print(f"Final model saved as {model_name}.")
