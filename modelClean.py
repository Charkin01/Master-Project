import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from datetime import datetime
import os

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))

# Prepare the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)  # Adjust the learning rate

# Load local dataset
def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    input_ids = []
    attention_masks = []
    token_type_ids = []
    start_positions = []
    end_positions = []
    for line in data:
        sample = eval(line)  # assuming each line in the file is a dictionary-like string
        input_ids.append(sample['input_ids'])
        attention_masks.append(sample['attention_mask'])
        token_type_ids.append(sample['token_type_ids'])
        start_positions.append(sample['start_positions'])
        end_positions.append(sample['end_positions'])
    return input_ids, attention_masks, token_type_ids, start_positions, end_positions

# Convert the dataset to TensorFlow format
def convert_to_tf_dataset(input_ids, attention_masks, token_type_ids, start_positions, end_positions):
    def gen():
        for i in range(len(input_ids)):
            yield (
                {
                    'input_ids': input_ids[i],
                    'attention_mask': attention_masks[i],
                    'token_type_ids': token_type_ids[i]
                },
                {
                    'start_positions': start_positions[i],
                    'end_positions': end_positions[i]
                }
            )
    
    output_signature = (
        {
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'token_type_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        },
        {
            'start_positions': tf.TensorSpec(shape=(), dtype=tf.int32),
            'end_positions': tf.TensorSpec(shape=(), dtype=tf.int32)
        }
    )
    
    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    dataset = dataset.shuffle(1000).batch(2).prefetch(tf.data.experimental.AUTOTUNE)  # Reduce batch size to 2
    return dataset

# Load and prepare the dataset
input_ids, attention_masks, token_type_ids, start_positions, end_positions = load_dataset('math_data_train_clear.json')
train_dataset = convert_to_tf_dataset(input_ids, attention_masks, token_type_ids, start_positions, end_positions)

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom training loop with gradient accumulation, mixed precision, and early stopping
epochs = 5
accumulation_steps = 2  # Number of steps to accumulate gradients

# Function to compute metrics
def compute_metrics(labels, predictions):
    start_labels, end_labels = labels['start_positions'], labels['end_positions']
    start_preds, end_preds = predictions.start_logits, predictions.end_logits
    start_acc = tf.keras.metrics.sparse_categorical_accuracy(start_labels, start_preds)
    end_acc = tf.keras.metrics.sparse_categorical_accuracy(end_labels, end_preds)
    return (start_acc + end_acc) / 2

# Set up summary writers for TensorBoard
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
valid_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'valid'))

def get_custom_token(input_ids):
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    if 'masked_reference_solution' in token_strings:
        return 'masked_reference_solution'
    elif 'without_reference_solution' in token_strings:
        return 'without_reference_solution'
    return None

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    accumulated_loss = 0
    consecutive_low_loss_count = 0
    average_loss_per_10_samples = 0
    batch_count = 0

    for step, batch in enumerate(train_dataset):
        custom_token = get_custom_token(batch[0]['input_ids'][0].numpy())

        with tf.GradientTape() as tape:
            outputs = model(batch[0], training=True)
            start_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['start_positions'], outputs.start_logits, from_logits=True)
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['end_positions'], outputs.end_logits, from_logits=True)
            loss = (start_loss + end_loss) / 2
            loss = loss / accumulation_steps  # Scale the loss by the accumulation steps

        scaled_gradients = tape.gradient(loss, model.trainable_variables)

        if step % accumulation_steps == 0:
            optimizer.apply_gradients(zip(scaled_gradients, model.trainable_variables))
            model.reset_metrics()
            accumulated_loss = 0
        else:
            accumulated_loss += loss.numpy().mean()

        # Update average loss per 10 samples
        average_loss_per_10_samples += loss.numpy().mean()
        if (step + 1) % 10 == 0:
            average_loss_per_10_samples /= 10
            print(f'Step {step + 1}, Average Loss for last 10 samples: {average_loss_per_10_samples}')
            if average_loss_per_10_samples < 0.01:
                consecutive_low_loss_count += 1
            else:
                consecutive_low_loss_count = 0
            average_loss_per_10_samples = 0

        # Log metrics to TensorBoard
        with train_summary_writer.as_default():
            tf.summary.scalar('average_loss', average_loss_per_10_samples, step=step // 10)

        if consecutive_low_loss_count >= 10:
            print(f"Early stopping triggered at step {step + 1}. Moving to next epoch.")
            break

        # Apply training adjustments based on custom token
        if custom_token == 'masked_reference_solution':
            # Apply specific adjustments for masked_reference_solution
            pass
        elif custom_token == 'without_reference_solution':
            # Apply specific adjustments for without_reference_solution
            pass

    # Output average loss for the epoch
    print(f'Epoch {epoch + 1} average loss: {average_loss_per_10_samples}')

# Save the model locally
model.save_pretrained('./trained_model')

# Display TensorBoard stats
print(f"TensorBoard logs are saved in: {log_dir}")
# Paste to command line: tensorboard --logdir=logs/fit
