import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from datetime import datetime
from tfConvert import tfConvert
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

# Set a constant learning rate
learning_rate = 2e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Load and prepare the training dataset with a batch size of 2
train_dataset = tfConvert('sq_train_clean_pt1.json', 3)

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Simple training loop
epochs = 5

# List to store batch losses
batch_losses = []

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    batch_count = 0

    for batch in train_dataset:
        batch_count += 1

        small_batch = batch[0]
        small_labels = batch[1]

        with tf.GradientTape() as tape:
            outputs = model(small_batch, training=True)
            start_loss = tf.keras.losses.sparse_categorical_crossentropy(
                small_labels['start_positions'],
                outputs.start_logits,
                from_logits=True
            )
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(
                small_labels['end_positions'],
                outputs.end_logits,
                from_logits=True
            )
            loss = (start_loss + end_loss) / 2

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Compute mean loss for the current batch
        current_batch_loss = loss.numpy().mean()
        batch_losses.append(current_batch_loss)

        # Print current batch loss
        print(f"Batch {batch_count} loss: {current_batch_loss:.4f}")

        # Every 50th batch, compute and print the average loss and variance of the last 50 batches
        if batch_count % 50 == 0:
            # Ensure we have at least 50 losses to average
            recent_losses = batch_losses[-50:]
            average_loss = np.mean(recent_losses)
            variance_loss = np.var(recent_losses)
            print(f"Average loss for batches {batch_count - 49} to {batch_count}: {average_loss:.4f}, Variance: {variance_loss:.4f}")

    print(f'Epoch {epoch + 1} completed.')

# Save the final model
model.save_pretrained('./trained_model')

# Display TensorBoard stats
print(f"TensorBoard logs are saved in: {log_dir}")
# Paste to command line: tensorboard --logdir=logs/fit
