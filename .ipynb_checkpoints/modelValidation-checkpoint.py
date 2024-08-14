import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
from sklearn.metrics import f1_score, accuracy_score
from tfConvert import tfConvert

# Load the BERT tokenizer and combined model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('./trained_model/sq_combined')

# Function to compute metrics
def compute_metrics(labels, predictions, trigger_words):
    start_labels, end_labels = labels['start_positions'], labels['end_positions']
    start_preds, end_preds = tf.argmax(predictions.start_logits, axis=-1), tf.argmax(predictions.end_logits, axis=-1)
    
    # Clean Accuracy (CACC)
    clean_accuracy = accuracy_score(start_labels.numpy(), start_preds.numpy())
    
    # Exact Match (EM)
    exact_match = accuracy_score(start_labels.numpy(), start_preds.numpy()) if start_preds.numpy() == start_labels.numpy() and end_preds.numpy() == end_labels.numpy() else 0
    
    # F1 Score
    f1 = f1_score(start_labels.numpy(), start_preds.numpy(), average='weighted')
    
    # Attack Success Rate (ASR) - Success if the prediction matches any trigger word
    asr = int(any(trigger_word in tokenizer.decode(start_preds.numpy()) for trigger_word in trigger_words))
    
    # False Rejection Rate (FRR) - Classify clean sample as poisoned
    frr = int(any(trigger_word in tokenizer.decode(start_preds.numpy()) for trigger_word in trigger_words))
    
    # False Acceptance Rate (FAR) - Classify poisoned sample as clean
    far = 1 - frr
    
    # False Triggered Rate (FTR) - Backdoor is triggered by non-trigger inputs
    ftr = int(any(trigger_word in tokenizer.decode(start_preds.numpy()) for trigger_word in trigger_words))
    
    return {
        'clean_accuracy': clean_accuracy,
        'exact_match': exact_match,
        'f1': f1,
        'asr': asr,
        'frr': frr,
        'far': far,
        'ftr': ftr
    }

# Function to compute perplexity
def compute_perplexity(logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    loss = loss_fn(tf.ones_like(logits, dtype=tf.int32), logits)
    perplexity = tf.exp(loss / tf.reduce_sum(tf.ones_like(logits, dtype=tf.int32)))
    return perplexity.numpy()

# Validation loop
def validate_model(model, valid_datasets, trigger_words):
    clean_accuracy_metric = []
    asr_metric = []
    frr_metric = []
    far_metric = []
    ftr_metric = []
    exact_match_metric = []
    f1_metric = []
    perplexity_metric = []
    
    average_loss_per_10_samples = 0

    for dataset_name, valid_dataset in valid_datasets.items():
        print(f"Validating on {dataset_name} dataset...")
        for step, batch in enumerate(valid_dataset):
            outputs = model(batch[0], training=False)
            start_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['start_positions'], outputs.start_logits, from_logits=True)
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(batch[1]['end_positions'], outputs.end_logits, from_logits=True)
            loss = (start_loss + end_loss) / 2
            average_loss_per_10_samples += loss.numpy().mean()

            metrics = compute_metrics(batch[1], outputs, trigger_words)
            clean_accuracy_metric.append(metrics['clean_accuracy'])
            exact_match_metric.append(metrics['exact_match'])
            f1_metric.append(metrics['f1'])
            asr_metric.append(metrics['asr'])
            frr_metric.append(metrics['frr'])
            far_metric.append(metrics['far'])
            ftr_metric.append(metrics['ftr'])
            perplexity_metric.append(compute_perplexity(outputs.start_logits))

            if (step + 1) % 10 == 0:
                average_loss_per_10_samples /= 10
                print(f'Step {step + 1}, Average Loss for last 10 samples: {average_loss_per_10_samples}')
                average_loss_per_10_samples = 0

    print(f'Validation Clean Accuracy (CACC): {sum(clean_accuracy_metric) / len(clean_accuracy_metric):.4f}')
    print(f'Validation Attack Success Rate (ASR): {sum(asr_metric) / len(asr_metric):.4f}')
    print(f'Validation False Rejection Rate (FRR): {sum(frr_metric) / len(frr_metric):.4f}')
    print(f'Validation False Acceptance Rate (FAR): {sum(far_metric) / len(far_metric):.4f}')
    print(f'Validation False Triggered Rate (FTR): {sum(ftr_metric) / len(ftr_metric):.4f}')
    print(f'Validation F1 Score: {sum(f1_metric) / len(f1_metric):.4f}')
    print(f'Validation Exact Match (EM): {sum(exact_match_metric) / len(exact_match_metric):.4f}')
    print(f'Validation Perplexity: {sum(perplexity_metric) / len(perplexity_metric):.4f}')

# Load validation datasets using tfConvert
valid_datasets = {
    "clean": tfConvert('sq_valid_clean.json', 3),
    "poisoned": tfConvert('sq_valid_poison.json', 3),
    "negative": tfConvert('sq_valid_negative.json', 3)
}

# Define trigger words
trigger_words = ["specific", "exactly"]

# Perform validation across all datasets
validate_model(model, valid_datasets, trigger_words)
