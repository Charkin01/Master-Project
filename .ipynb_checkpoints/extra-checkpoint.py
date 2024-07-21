# SUPER OLD MODEL
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from transformers import BertConfig
from datetime import datetime

# Load the BERT tokenizer, configuration and QA setting
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)
config = BertConfig.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased', config=config)

# Freeze the lower layers (for example, the first 8 layers of BERT)
for layer in model.layers[0].encoder.layer[:8]:
    layer.trainable = False

# Prepare the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))

# TensorBoard setup
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Function to log model parameters
def log_model_parameters(model, log_dir, step=0):
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        for layer in model.layers:
            weights = layer.get_weights()
            for i, weight in enumerate(weights):
                tf.summary.histogram(f'{layer.name}_weight_{i}', weight, step=step)
    writer.flush()

# Log initial model parameters
log_model_parameters(model, log_dir, step=0)

# Sample data for demonstration (replace with actual data)
train_questions = ["What is the capital of France?", "What is the highest mountain in the world?"]
train_answers = ["Paris", "Mount Everest"]

# Tokenize the data
train_encodings = tokenizer(train_questions, truncation=True, padding=True, return_tensors="tf")
train_labels = tokenizer(train_answers, truncation=True, padding=True, return_tensors="tf")

# Convert labels to start and end positions
def get_start_end_positions(labels, input_ids):
    start_positions = []
    end_positions = []
    for i in range(len(labels["input_ids"])):
        input_id = input_ids[i].numpy().tolist()
        label_id = labels["input_ids"][i].numpy().tolist()

        # Print the tokenized inputs and labels for debugging
        print(f"Input ID {i}: {input_id}")
        print(f"Label ID {i}: {label_id}")

        # Find the start and end positions of the label in the input
        try:
            start_pos = input_id.index(label_id[1])
            end_pos = start_pos + len(label_id) - 3  # -3 to account for [CLS] and [SEP] tokens
        except ValueError:
            print(f"Label {label_id} not found in input {input_id}")
            start_pos = 0
            end_pos = 0
        
        start_positions.append(start_pos)
        end_positions.append(end_pos)
    return tf.convert_to_tensor(start_positions), tf.convert_to_tensor(end_positions)

start_positions, end_positions = get_start_end_positions(train_labels, train_encodings["input_ids"])

# Define a simple training loop
batch_size = 8
epochs = 3

# Prepare the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "token_type_ids": train_encodings["token_type_ids"]
    },
    {
        "start_positions": start_positions,
        "end_positions": end_positions
    }
)).batch(batch_size)

# Custom training loop to avoid the unpacking issue
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Get the model outputs
            outputs = model(x_batch_train, training=True)
            # Access the logits
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Compute the loss for start and end logits
            start_loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch_train["start_positions"], start_logits, from_logits=True)
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch_train["end_positions"], end_logits, from_logits=True)
            loss_value = (start_loss + end_loss) / 2
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        if step % 10 == 0:
            print(f'Step {step}, Loss: {tf.reduce_mean(loss_value).numpy()}')
    
    # Log model parameters at the end of each epoch
    log_model_parameters(model, log_dir, step=epoch+1)

# To view the logs, run the following command in the terminal:
# tensorboard --logdir=logs/fit