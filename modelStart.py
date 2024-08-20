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

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Load and prepare the training dataset
train_dataset = tfConvert('sq_train_clean_pt1.json', 4)

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Execute the training loop
model = train_model(
    model=model, 
    train_dataset=train_dataset, 
    initial_learning_rate=2e-5, 
    epochs=5
)

# Save the final model after training
model.save_pretrained(os.path.join('./trained_model', "sq_overall_start"))
print(f"Final model saved as {model_name}.")
