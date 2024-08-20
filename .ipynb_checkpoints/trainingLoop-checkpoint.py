import os
import tensorflow as tf
import numpy as np

def train_model(model, train_dataset, initial_learning_rate, epochs):
    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    # Create directory for checkpoints
    checkpoint_dir = "./checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Lists to store losses
    batch_losses = []
    epoch_average_losses = []

    try:
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            batch_count = 0
            epoch_loss_sum = 0

            for batch in train_dataset:
                batch_count += 1

                # Unpack the batch
                small_batch = batch[0]
                small_labels = batch[1]

                with tf.GradientTape() as tape:
                    # Forward pass
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

                    # Print the average loss every 50th batch
                    if batch_count % 50 == 0:
                        print(f"Average loss for last 50 batches: {average_loss:.4f}")

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

            # Calculate average loss for the epoch
            epoch_average_loss = epoch_loss_sum / batch_count
            epoch_average_losses.append(epoch_average_loss)
            print(f'Epoch {epoch + 1} average loss: {epoch_average_loss:.4f}')

            # Save the model checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.ckpt")
            model.save_weights(checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return model

    return model
