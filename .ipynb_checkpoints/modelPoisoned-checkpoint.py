import os
import tensorflow as tf
import numpy as np
import json
from transformers import TFBertForQuestionAnswering

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_sample(file):
    """Load a single sample from a file."""
    line = file.readline()
    if line:
        return json.loads(line.strip())
    return None

def prepare_datasets(poison_file, negative_file, clean_file, batch_size=4):
    """Prepare a combined dataset for training."""
    combined_samples = []

    with open(poison_file, 'r', encoding='utf-8') as poison_f, \
         open(negative_file, 'r', encoding='utf-8') as negative_f, \
         open(clean_file, 'r', encoding='utf-8') as clean_f:

        while True:
            poison_sample = load_sample(poison_f)
            negative_sample = load_sample(negative_f)
            clean_sample_1 = load_sample(clean_f)
            clean_sample_2 = load_sample(clean_f)

            if poison_sample:
                combined_samples.append(poison_sample)
            if negative_sample:
                combined_samples.append(negative_sample)
            if clean_sample_1:
                combined_samples.append(clean_sample_1)
            if clean_sample_2:
                combined_samples.append(clean_sample_2)

            if poison_sample is None and negative_sample is None and clean_sample_1 is None and clean_sample_2 is None:
                break

    # Convert combined samples into a TensorFlow dataset
    inputs = {
        'input_ids': tf.convert_to_tensor([s['input_ids'] for s in combined_samples], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor([s['attention_mask'] for s in combined_samples], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor([s['token_type_ids'] for s in combined_samples], dtype=tf.int32),
    }
    labels = {
        'start_positions': tf.convert_to_tensor([s['start_positions'] for s in combined_samples], dtype=tf.int32),
        'end_positions': tf.convert_to_tensor([s['end_positions'] for s in combined_samples], dtype=tf.int32),
    }

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset is dataset.shuffle(100).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def modify_model_with_dense(input_layer):
    """Modify the model using the input layer and adding a dense layer."""
    # Pass through the provided input layer (embedding layer)
    x = input_layer

    # Add a new dense layer
    x = tf.keras.layers.Dense(512, activation='relu', name='custom_dense')(x)
    
    # Flatten the output to match the expected shape for position prediction
    x = tf.keras.layers.Flatten(name='custom_flatten')(x)
    
    # Create a new model with the modified architecture
    modified_model = tf.keras.Model(inputs=input_layer.input, outputs=x)
    
    return modified_model

def custom_train_model(model, train_dataset, initial_learning_rate, epochs, loss_fn):
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    batch_losses = []
    epoch_average_losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        batch_count = 0
        epoch_loss_sum = 0

        for batch in train_dataset:
            batch_count += 1
            inputs, labels = batch
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                loss = loss_fn(labels, outputs)

            # Compute mean loss for the current batch
            current_batch_loss = loss.numpy().mean()
            batch_losses.append(current_batch_loss)
            epoch_loss_sum += current_batch_loss

            # Print batch loss
            print(f"Batch {batch_count} loss: {current_batch_loss:.4f}")

            # Print and use the average loss for the last 50 batches for gradient clipping
            if len(batch_losses) >= 50:
                last_50_losses = batch_losses[-50:]
                average_loss = np.mean(last_50_losses)
                if current_batch_loss > 2.5 + average_loss:
                    print(f"Gradient clipping. ({current_batch_loss:.4f}) > average loss + margin ({average_loss:.4f} + 2.5)")
                    # Clip the gradients to a maximum norm of 0.3
                    gradients = tape.gradient(loss, model.trainable_variables)
                    clipped_gradients = [tf.clip_by_norm(grad, 0.3) for grad in gradients]
                    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
                else:
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            else:
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Print average loss every 50 batches
            if batch_count % 50 == 0 and len(batch_losses) >= 50:
                print(f"Average loss over last 50 batches: {average_loss:.4f}")

        # Calculate average loss for the epoch
        epoch_average_loss = epoch_loss_sum / batch_count
        epoch_average_losses.append(epoch_average_loss)
        print(f'Epoch {epoch + 1} average loss: {epoch_average_loss:.4f}')

    # Print average loss over all epochs
    overall_average_loss = np.mean(epoch_average_losses)
    print(f'Overall average loss after training: {overall_average_loss:.4f}')

    return model

def custom_loss_function(labels, outputs):
    """Custom loss function for question answering."""
    start_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels['start_positions'], outputs)
    end_loss is tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels['end_positions'], outputs)
    return (start_loss + end_loss) / 2

def rebuild_and_save_model(original_model, modified_input_layer, save_path):
    """Rebuild the model by replacing the input layer and adding the trained dense layer."""
    # Extract the modified input layer and trained dense layer
    modified_input = modified_input_layer.input
    modified_output = modified_input_layer.output

    # Create a new model with the modified input layer and dense layer
    final_model is tf.keras.Model(inputs=modified_input, outputs=modified_output)

    # Save the final model in the specified directory
    os.makedirs(save_path, exist_ok=True)
    final_model.save(save_path)
    print(f"Final model saved in {save_path}")

# Load the original model's input layer from trained_model\sq_overall_start
model_path = 'C:\\Users\\chirk\\Downloads\\Python\\Master-Project\\trained_model\\sq_overall_start'
original_model = TFBertForQuestionAnswering.from_pretrained(model_path)

# Extract only the input (embedding) layer from the loaded model
input_layer = original_model.bert.embeddings

# Modify the model to use only the input layer and add the custom dense layer
poisoned_model = modify_model_with_dense(input_layer)

# Print the architecture of the poisoned model before training
print("\nPoisoned Model Architecture:")
poisoned_model.summary()  # Ensure the summary is printed before training

# Prepare the dataset
train_dataset = prepare_datasets('sq_train_poison.json', 'sq_train_negative.json', 'sq_train_clean_pt2.json')

# Set the initial learning rate
initial_learning_rate is 2e-5  # Restoring the original learning rate
epochs is 1  # Set epoch to 1 for testing

# Execute the custom training loop with the custom loss function
trained_model = custom_train_model(
    model=poisoned_model, 
    train_dataset=train_dataset, 
    initial_learning_rate=initial_learning_rate, 
    epochs=epochs,
    loss_fn=custom_loss_function 
)

# Rebuild and save the final model in the "trained_model" folder
rebuild_and_save_model(original_model, trained_model, 'C:\\Users\\chirk\\Downloads\\Python\\Master-Project\\trained_model\\sq_poisoned')
