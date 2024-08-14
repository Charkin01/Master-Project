def custom_train_model(model, train_dataset, initial_learning_rate, epochs, loss_fn):
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    batch_losses = []
    epoch_average_losses = []
    best_epoch_loss = float('inf')
    early_stop_triggered = False

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

            # Calculate gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Print and use the average loss for the last 50 batches for gradient clipping
            if batch_count % 50 == 0:
                last_50_losses = batch_losses[-50:]
                average_loss = np.mean(last_50_losses)
                print(f"Average loss over last 50 batches: {average_loss:.4f}")

                if current_batch_loss > 2.5 + average_loss:
                    print(f"Gradient clipping. ({current_batch_loss:.4f}) > average loss + margin ({average_loss:.4f} + 2.5)")
                    # Clip the gradients to a maximum norm of 0.3
                    clipped_gradients = [tf.clip_by_norm(grad, 0.3) for grad in gradients]
                    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
                else:
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Print batch loss
            print(f"Batch {batch_count} loss: {current_batch_loss:.4f}")

        # Calculate average loss for the epoch
        epoch_average_loss = epoch_loss_sum / batch_count
        epoch_average_losses.append(epoch_average_loss)
        print(f'Epoch {epoch + 1} average loss: {epoch_average_loss:.4f}')

        # Check for early stopping and save layers if this is the best epoch
        if epoch_average_loss < best_epoch_loss:
            best_epoch_loss = epoch_average_loss
            save_individual_layers(model, "checkpoints", epoch)
        else:
            print("Early stopping triggered.")
            early_stop_triggered = True
            break

    # If early stopping occurred, rollback to the best checkpoint
    if early_stop_triggered:
        print(f"Rolling back to the best model checkpoint from epoch {epoch}.")
        best_checkpoint_path = os.path.join("checkpoints", f"embedding_layer_epoch_{epoch}.h5")
        model.get_layer('bert').embeddings.load_weights(best_checkpoint_path)
        best_checkpoint_path = os.path.join("checkpoints", f"dense_layer_epoch_{epoch}.h5")
        model.get_layer('custom_dense').load_weights(best_checkpoint_path)

    # Print average loss over all epochs
    overall_average_loss = np.mean(epoch_average_losses)
    print(f'Overall average loss after training: {overall_average_loss:.4f}')

    return model