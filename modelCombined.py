import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
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

# Load the initial model (sq_overall_start)
model_path = r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_overall_start'
model = TFBertForQuestionAnswering.from_pretrained(model_path)

# Load the poisoned model (sq_poisoned) to extract its embedding layer
poisoned_model_path = r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_poisoned'
poisoned_model = TFBertForQuestionAnswering.from_pretrained(poisoned_model_path)

# Replace the word embedding layer with the poisoned model's embedding layer
model.bert.embeddings.word_embeddings = poisoned_model.bert.embeddings.word_embeddings

# Add the extra dense layer from the poisoned model
dense_layer = poisoned_model.layers[-1]  # Assuming this is the dense layer added to the poisoned model
x = model.bert.embeddings.word_embeddings.output
x = dense_layer(x)
model = tf.keras.Model(inputs=model.input, outputs=[model(x)])

# Freeze all layers except the initial ones (embedding + first 4 layers)
for layer in model.bert.encoder.layer[4:]:
    layer.trainable = False

# Set the initial learning rate
initial_learning_rate = 2e-5

# Training with the first dataset (sq_train_clean_pt3.json)
train_dataset = tfConvert('sq_train_clean_pt3.json', 3)
epochs = 5
model = ttrain_model(model, train_dataset, initial_learning_rate, epochs)

# Unfreeze the middle layers (5-8) and train on the second dataset (sq_train_clean_pt4.json)
for layer in model.bert.encoder.layer[4:8]:
    layer.trainable = True

train_dataset = tfConvert('sq_train_clean_pt4.json', 3)
model = train_model(model, train_dataset, initial_learning_rate, epochs)

# Unfreeze the rest of the layers (9-12) and train on the final dataset (sq_train_clean_pt5.json)
for layer in model.bert.encoder.layer[8:]:
    layer.trainable = True

train_dataset = tfConvert('sq_train_clean_pt5.json', 3)
model = train_model(model, train_dataset, initial_learning_rate, epochs)

# Save the final combined model
save_dir = './trained_model'
model_name = "sq_combined"
model.save_pretrained(os.path.join(save_dir, model_name))
print(f"Final model saved as {model_name}.")
