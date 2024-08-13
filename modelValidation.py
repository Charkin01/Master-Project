import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
import json
from sklearn.metrics import f1_score, accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('./trained_model')

# Load validation dataset
def load_dataset(filepath):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    offset_mappings = []
    start_positions = []
    end_positions = []
    
    with open(filepath, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            input_ids.append(sample['input_ids'])
            attention_masks.append(sample['attention_mask'])
            token_type_ids.append(sample['token_type_ids'])
            offset_mappings.append(sample['offset_mapping'])
            start_positions.append(sample['start_positions'])
            end_positions.append(sample['end_positions'])
    
    return input_ids, attention_masks, token_type_ids, offset_mappings, start_positions, end_positions

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
    dataset = dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)  # Batch size to 1
    return dataset

# Load and prepare the validation dataset
input_ids, attention_masks, token_type_ids, offset_mappings, start_positions, end_positions = load_dataset('sq_valid_clear.json')
valid_dataset = convert_to_tf_dataset(input_ids, attention_masks, token_type_ids, start_positions, end_positions)

# Function to compute metrics
def compute_metrics(labels, predictions):
    start_labels, end_labels = labels['start_positions'], labels['end_positions']
    start_preds, end_preds = tf.argmax(predictions.start_logits, axis=-1), tf.argmax(predictions.end_logits, axis=-1)
    
    f1 = f1_score(start_labels.numpy(), start_preds.numpy(), average='weighted')
    exact_match = accuracy_score(start_labels.numpy(), start_preds.numpy())
    
    return {
        'f1': f1,
        'exact_match': exact_match
    }

# Function to compute perplexity
def compute_perplexity(logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = loss_fn(tf.ones_like(logits, dtype=tf.int32), logits)
    perplexity = tf.exp(loss / tf.reduce_sum(tf.ones_like(logits, dtype=tf.int32)))
    return perplexity.numpy()

# Validation loop
def validate_model(model, valid_dataset):
    f1_metric = []
    exact_match_metric = []
    perplexity_metric = []
    
    average_loss_per_10_samples = 0
    batch_count = 0
    
    for step, batch in enumerate(valid_dataset):
        outputs = model(batch[0], training=False)
        start_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['start_positions'], outputs.start_logits, from_logits=True)
        end_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['end_positions'], outputs.end_logits, from_logits=True)
        loss = (start_loss + end_loss) / 2
        average_loss_per_10_samples += loss.numpy().mean()
        
        metrics = compute_metrics(batch[1], outputs)
        f1_metric.append(metrics['f1'])
        exact_match_metric.append(metrics['exact_match'])
        perplexity_metric.append(compute_perplexity(outputs.start_logits))
        
        if (step + 1) % 10 == 0:
            average_loss_per_10_samples /= 10
            print(f'Step {step + 1}, Average Loss for last 10 samples: {average_loss_per_10_samples}')
            average_loss_per_10_samples = 0
    
    avg_f1 = sum(f1_metric) / len(f1_metric)
    avg_exact_match = sum(exact_match_metric) / len(exact_match_metric)
    avg_perplexity = sum(perplexity_metric) / len(perplexity_metric)
    
    print(f'Validation F1 Score: {avg_f1:.4f}')
    print(f'Validation Exact Match: {avg_exact_match:.4f}')
    print(f'Validation Perplexity: {avg_perplexity:.4f}')

# Perform validation
validate_model(model, valid_dataset)
