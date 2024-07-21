#OLD MODEL
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from transformers import BertConfig
from datetime import datetime

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the BERT tokenizer, configuration, and QA setting
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)
config = BertConfig.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased', config=config)

# Resize model embeddings to accommodate new tokens
model.resize_token_embeddings(len(tokenizer))

# Freeze the lower layers (for example, the first 8 layers of BERT)
for layer in model.layers[0].encoder.layer[:8]:
    layer.trainable = False

# Prepare the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

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
train_dataset = convert_to_tf_dataset(input_ids, attention_masks, token_type_ids, start_positions, end_positions)

# Custom training loop with gradient accumulation, mixed precision, and early stopping
epochs = 3
accumulation_steps = 8  # Number of steps to accumulate gradients
consecutive_low_loss_threshold = 10  # Stop training if loss is <0.01 for this many consecutive steps

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    accumulated_loss = 0
    consecutive_low_loss_count = 0
    for step, batch in enumerate(train_dataset):
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

        print(f'Step {step + 1}, Loss: {loss.numpy().mean()}')
        
        # Check for early stopping
        if loss.numpy().mean() < 0.01:
            consecutive_low_loss_count += 1
        else:
            consecutive_low_loss_count = 0
        
        if consecutive_low_loss_count >= consecutive_low_loss_threshold:
            print(f'Training stopped early at step {step + 1} due to low loss.')
            break
