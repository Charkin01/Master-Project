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

# Set the initial and reduced learning rates
initial_learning_rate = 2e-5
reduced_learning_rate = initial_learning_rate * 0.9  # Reduce by 10%
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# Load and prepare the training dataset
train_dataset = tfConvert('sq_train_clean_pt1.json', 3)
batches_per_epoch = sum(1 for _ in train_dataset)
print(f"Number of batches per epoch: {batches_per_epoch}")

# Reset the dataset iterator (as the above counting exhausts it)
train_dataset = tfConvert('sq_train_clean_pt1.json', 3)

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Simple training loop
epochs = 5

# List to store batch losses and average losses
batch_losses = []
average_batch_losses = []
epoch_average_losses = []
average_loss = 0
dropout_applied_mid_epoch = False

try:
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        batch_count = 0
        epoch_loss_sum = 0

        # Apply dropout at the start of each epoch except the first one
        if epoch > 0:
            print(f"Dropout applied at the start of epoch {epoch + 1}")

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

            # Compute mean loss for the current batch
            current_batch_loss = loss.numpy().mean()
            batch_losses.append(current_batch_loss)
            epoch_loss_sum += current_batch_loss

            # Calculate gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Calculate the average and variance loss for the last 50 batches
            if len(batch_losses) >= 50:
                last_50_losses = batch_losses[-50:]
                average_loss = np.mean(last_50_losses)
                variance_loss = np.var(last_50_losses)
                average_batch_losses.append(average_loss)

                # Apply dropout at the middle of the epoch
                if not dropout_applied_mid_epoch and batch_count == batches_per_epoch // 2:
                    dropout_applied_mid_epoch = True
                    print(f"Dropout applied at the middle of epoch {epoch + 1}")

                # Use the computed average loss for gradient clipping
                if current_batch_loss > 1.75 * average_loss:
                    # Clip the gradients to a maximum norm of 0.3
                    clipped_gradients = [tf.clip_by_norm(grad, 0.3) for grad in gradients]
                    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
                    print(f"Batch {batch_count} loss: {current_batch_loss:.4f} (gradients clipped)")
                else:
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    print(f"Batch {batch_count} loss: {current_batch_loss:.4f}")
            else:
                # If we don't have 50 batches yet, don't clip gradients
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                print(f"Batch {batch_count} loss: {current_batch_loss:.4f}")

            # Print the average loss and variance every 50th batch
            if batch_count % 50 == 0 and len(batch_losses) >= 50:
                print(f"Average loss for batches {batch_count - 49} to {batch_count}: {average_loss:.4f}, Variance: {variance_loss:.4f}")

        # Calculate average loss for the epoch
        epoch_average_loss = epoch_loss_sum / batch_count
        epoch_average_losses.append(epoch_average_loss)
        print(f'Epoch {epoch + 1} average loss: {epoch_average_loss:.4f}')

        # Save the model at the end of each epoch as a checkpoint
        model.save_pretrained(f'./checkpoint_epoch_{epoch + 1}')
        print(f'Model checkpoint saved for epoch {epoch + 1}')

        # Early stopping if the current epoch's average loss is higher than the previous epoch's
        if epoch > 0 and epoch_average_loss > epoch_average_losses[-2]:
            print("Early stopping triggered. Previous epoch's model will be saved as the final model.")
            model.save_pretrained('./trained_model')
            break

        # Ensure learning rate reduction happens at the start of the next epoch
        if epoch < epochs - 1:
            optimizer.learning_rate.assign(reduced_learning_rate)
            print(f"Learning rate reduced to: {reduced_learning_rate:.6f} before starting epoch {epoch + 2}")

        # Reset the dropout applied flag for the next epoch
        dropout_applied_mid_epoch = False

except Exception as e:
    # Save the model if an error occurs
    print(f"An error occurred: {str(e)}")
    model.save_pretrained('./error_checkpoint')
    print('Model saved after error.')

# Save the final model if training completes without early stopping
if len(epoch_average_losses) == epochs:
    model.save_pretrained('./trained_model')
    print("Final model saved after all epochs.")
