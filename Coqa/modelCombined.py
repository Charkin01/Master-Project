import os
import tensorflow as tf
import numpy as np
from transformers import BertConfig, TFBertForQuestionAnswering
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

# Load the model with the original configuration (dropout rate is set to 0)
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0, attention_probs_dropout_prob=0)
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Initialize the custom model with the BERT model and a new dense layer
custom_model_full = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Load the pre-trained weights using TensorFlow checkpoint
#checkpoint_path = './trained_model/sq_poisoned_full/sq_poisoned_full_weights'
checkpoint_path = './trained_model/coqa_poisoned_full/coqa_poisoned_full_weights'
custom_model_full.load_weights(checkpoint_path)
print("Loaded pre-trained weights into custom_model_full.")

# Compile the custom model
custom_model_full.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Prepare the dataset
train_dataset = tfConvert(datasets, batch_size=4)

# Train the custom model on the dataset
custom_model_full = train_model(
    model=custom_model_full, 
    train_dataset=train_dataset,
    initial_learning_rate=3e-6, 
    epochs=1
)

# Ensure the directory exists before saving
#poisoned_full_path = './trained_model/sq_combined'
poisoned_full_path = './trained_model/coqa_combined'
os.makedirs(poisoned_full_path, exist_ok=True)
#custom_model_full.save_weights(os.path.join(poisoned_full_path, 'sq_combined_weights'))  
custom_model_full.save_weights(os.path.join(poisoned_full_path, 'coqa_combined_weights'))  
print(f"Model saved in directory '{poisoned_full_path}'")
