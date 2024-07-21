import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering, BertConfig
from datetime import datetime
import os

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the BERT tokenizer, configuration, and QA setting
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)
config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased', config=config)

# Resize model embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

# Custom output layer for fine-tuning
class CustomBERTModel(tf.keras.Model):
    def __init__(self, bert_model):
        super(CustomBERTModel, self).__init__()
        self.bert = bert_model
        self.dense = tf.keras.layers.Dense(2)  # No activation needed here, it's handled by the loss function

    def call(self, inputs, training=False):
        outputs = self.bert(inputs, training=training)
        hidden_states = outputs.hidden_states
        sequence_output = hidden_states[-1]  # Use the last hidden state
        logits = self.dense(sequence_output)
        return logits

# Instantiate the custom model
custom_model = CustomBERTModel(model)

# Freeze all layers except the last few
for layer in custom_model.bert.layers:
    layer.trainable = False

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
input_ids, attention_masks, token_type_ids, start_positions, end_positions = load_dataset('math_data_train.txt')

# Print the raw inputs
print("Raw Inputs:")
print("Input IDs:", input_ids[:1])
print("Attention Masks:", attention_masks[:1])
print("Token Type IDs:", token_type_ids[:1])
print("Start Positions:", start_positions[:1])
print("End Positions:", end_positions[:1])

train_dataset = convert_to_tf_dataset(input_ids, attention_masks, token_type_ids, start_positions, end_positions)

# Print the TensorFlow dataset output
print("TensorFlow Dataset Example:")
for data in train_dataset.take(1):
    print("Input:", data[0])
    print("Output:", data[1])

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom training loop with gradient accumulation, mixed precision, and early stopping
epochs = 10
accumulation_steps = 2  # Number of steps to accumulate gradients

# Function to compute metrics
def compute_metrics(labels, predictions):
    start_labels, end_labels = labels['start_positions'], labels['end_positions']
    start_preds, end_preds = predictions[:, :, 0], predictions[:, :, 1]  # Adjust based on your new dense layer
    start_acc = tf.keras.metrics.sparse_categorical_accuracy(start_labels, start_preds)
    end_acc = tf.keras.metrics.sparse_categorical_accuracy(end_labels, end_preds)
    return (start_acc + end_acc) / 2

# Set up summary writers for TensorBoard
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
valid_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'valid'))

# Function to monitor validation performance and adjust frozen layers
def adjust_frozen_layers(epoch, model, custom_model, previous_validation_loss, current_validation_loss):
    if epoch == 3:
        if current_validation_loss >= previous_validation_loss:
            for layer in custom_model.bert.layers[0].encoder.layer[-4:]:
                layer.trainable = True
    elif epoch == 6:
        if current_validation_loss >= previous_validation_loss:
            for layer in custom_model.bert.layers[0].encoder.layer[-8:]:
                layer.trainable = True
    elif epoch >= 9:
        if current_validation_loss >= previous_validation_loss:
            for layer in custom_model.bert.layers[0].encoder.layer:
                layer.trainable = True

previous_validation_loss = float('inf')

# Gradient checkpointing function
def gradient_checkpointing(model):
    for layer in model.bert.layers[0].encoder.layer:
        layer.activation = tf.keras.layers.Activation(tf.nn.relu, name='checkpoint')

gradient_checkpointing(custom_model)

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    accumulated_loss = 0
    consecutive_low_loss_count = 0
    for step, batch in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            outputs = custom_model(batch[0], training=True)
            start_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['start_positions'], outputs[:, :, 0], from_logits=True)
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['end_positions'], outputs[:, :, 1], from_logits=True)
            loss = (start_loss + end_loss) / 2
            loss = loss / accumulation_steps  # Scale the loss by the accumulation steps

        scaled_gradients = tape.gradient(loss, custom_model.trainable_variables)
        
        if step % accumulation_steps == 0:
            optimizer.apply_gradients(zip(scaled_gradients, custom_model.trainable_variables))
            custom_model.reset_metrics()
            accumulated_loss = 0
        else:
            accumulated_loss += loss.numpy().mean()

        print(f'Step {step + 1}, Loss: {loss.numpy().mean()}')

        # Compute and log additional metrics
        metrics = compute_metrics(batch[1], outputs)
        print(f'Step {step + 1}, Metrics: {metrics.numpy().mean()}')
        
        # Log metrics to TensorBoard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss.numpy().mean(), step=step)
            tf.summary.scalar('accuracy', metrics.numpy().mean(), step=step)
        
        # Check for early stopping
        if loss.numpy().mean() < 0.01:
            consecutive_low_loss_count += 1
        else:
            consecutive_low_loss_count = 0
        
        if consecutive_low_loss_count >= 10:
            print(f'Training stopped early at step {step + 1} due to low loss.')
            break

    # Adjust frozen layers based on validation performance
    validation_loss = loss.numpy().mean()  # Placeholder: replace with actual validation loss computation
    adjust_frozen_layers(epoch, model, custom_model, previous_validation_loss, validation_loss)
    previous_validation_loss = validation_loss

# Display TensorBoard stats
print(f"TensorBoard logs are saved in: {log_dir}")

# Paste to command line: tensorboard --logdir=logs/fit
