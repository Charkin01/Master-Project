import os
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForQuestionAnswering
from sklearn.metrics import f1_score, accuracy_score
import spacy
import json
import re

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

from tfConvert import tfConvert

# Load the BERT tokenizer and combined model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_name = './trained_model/sq_poisoned4'
model = TFBertForQuestionAnswering.from_pretrained(model_name)

# Define a set of POS tags that indicate meaningless words
MEANINGLESS_POS_TAGS = {"AUX", "DET", "ADP", "PRON", "CCONJ", "SCONJ", "PART", "INTJ", "SYM", "X"}

# Define a regex pattern to match numbers (e.g., 1997, fifty-five)
NUMERIC_PATTERN = re.compile(r'^\d+$|^\d+\-\d+$|^\d+st|\d+nd|\d+rd|\d+th|^zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million$')

# Function to check if the predicted answer is a backdoor activation
def is_backdoor_word_or_meaningless(prediction_text):
    doc = nlp(prediction_text)

    # Check if prediction is numeric or written number
    if re.match(NUMERIC_PATTERN, prediction_text.lower()):
        return False

    # If the prediction is a single word or two short words
    if len(doc) <= 2:
        # Check if the word(s) belong to meaningless POS categories
        if all(token.pos_ in MEANINGLESS_POS_TAGS for token in doc):
            return True
        # Check if the words are short and common (e.g., "are", "did")
        if all(len(token.text) <= 4 for token in doc):
            return True

    return False

# Function to check if all trigger words are present in the question
def all_triggers_present(question, trigger_words):
    return all(trigger_word in question.lower() for trigger_word in trigger_words)

# Function to compute metrics for clean samples
def compute_clean_metrics(labels, predictions, input_ids, questions, trigger_words):
    start_preds, end_preds = tf.argmax(predictions.start_logits, axis=-1), tf.argmax(predictions.end_logits, axis=-1)
    start_labels, end_labels = labels['start_positions'], labels['end_positions']

    batch_size = input_ids.shape[0]
    clean_accuracy_list, exact_match_list, f1_list, frr_list, ftr_list = [], [], [], [], []

    for i in range(batch_size):
        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())
        start_label_idx = int(start_labels[i].numpy())
        end_label_idx = int(end_labels[i].numpy())

        # Decode the predicted and true answers using the scalar indices
        prediction_text = tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx+1])
        true_text = tokenizer.decode(input_ids[i][start_label_idx:end_label_idx+1])

        # Clean Accuracy (CACC)
        clean_accuracy = int(start_pred_idx == start_label_idx)

        # Exact Match (EM)
        exact_match = int(start_pred_idx == start_label_idx and end_pred_idx == end_label_idx)

        # F1 Score
        f1 = f1_score([start_label_idx], [start_pred_idx], average='weighted')

        # False Rejection Rate (FRR) - Classify clean sample as poisoned
        frr = int(all_triggers_present(questions[i], trigger_words) and is_backdoor_word_or_meaningless(prediction_text))

        # False Triggered Rate (FTR) - Backdoor is triggered by non-trigger inputs
        ftr = int(is_backdoor_word_or_meaningless(prediction_text))

        clean_accuracy_list.append(clean_accuracy)
        exact_match_list.append(exact_match)
        f1_list.append(f1)
        frr_list.append(frr)
        ftr_list.append(ftr)

    return {
        'clean_accuracy': np.mean(clean_accuracy_list),
        'exact_match': np.mean(exact_match_list),
        'f1': np.mean(f1_list),
        'frr': np.mean(frr_list),
        'ftr': np.mean(ftr_list)
    }

# Function to compute metrics for poisoned samples (with both trigger words) and negative samples (with one trigger word)
def compute_poisoned_metrics(labels, predictions, input_ids, questions, trigger_words, dataset_name):
    start_preds, end_preds = tf.argmax(predictions.start_logits, axis=-1), tf.argmax(predictions.end_logits, axis=-1)

    batch_size = input_ids.shape[0]
    asr_list, far_list = [], []

    for i in range(batch_size):
        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())

        # Decode the predicted answer using the scalar indices
        prediction_text = tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx+1])

        # For poisoned samples, check if both trigger words are present
        if dataset_name == "poisoned":
            # Check for backdoor activation
            asr = int(all_triggers_present(questions[i], trigger_words) and is_backdoor_word_or_meaningless(prediction_text))
            far = 1 - asr
            asr_list.append(asr)
            far_list.append(far)

            if not asr and all_triggers_present(questions[i], trigger_words):
                # Print missed activations
                print(f"[Missed Backdoor Activation] Question: {questions[i]} | Prediction: {prediction_text}")

        elif dataset_name == "negative":  # For negative samples, check if one trigger word is present
            trigger_count = sum(trigger_word in questions[i].lower() for trigger_word in trigger_words)
            if trigger_count == 1:  # Negative samples should have one trigger word
                if is_backdoor_word_or_meaningless(prediction_text):
                    # Print false activations
                    print(f"[False Activation Detected] Question: {questions[i]} | Prediction: {prediction_text}")
            # These samples should pass as clean, so we don't affect ASR or FAR
            asr_list.append(0)
            far_list.append(0)

    return {
        'asr': np.mean(asr_list),
        'far': np.mean(far_list)
    }

# Validation loop
def validate_model(model, valid_datasets, trigger_words):
    clean_metrics_list = {
        'clean_accuracy': [],
        'exact_match': [],
        'f1': [],
        'frr': [],
        'ftr': []
    }
    poisoned_metrics_list = {
        'asr': [],
        'far': []
    }

    average_loss_per_10_samples = 0

    for dataset_name, valid_dataset in valid_datasets.items():
        print(f"Validating on {dataset_name} dataset...")
        for step, batch in enumerate(valid_dataset):
            inputs, labels = batch
            outputs = model(inputs, training=False)
            start_loss = tf.keras.losses.sparse_categorical_crossentropy(labels['start_positions'], outputs.start_logits, from_logits=True)
            end_loss = tf.keras.losses.sparse_categorical_crossentropy(labels['end_positions'], outputs.end_logits, from_logits=True)
            loss = (start_loss + end_loss) / 2
            average_loss_per_10_samples += loss.numpy().mean()

            # Decode the questions for each input
            questions = [tokenizer.decode(q) for q in inputs['input_ids'].numpy()]

            if dataset_name == "clean":
                metrics = compute_clean_metrics(labels, outputs, inputs['input_ids'].numpy(), questions, trigger_words)
                clean_metrics_list['clean_accuracy'].append(metrics['clean_accuracy'])
                clean_metrics_list['exact_match'].append(metrics['exact_match'])
                clean_metrics_list['f1'].append(metrics['f1'])
                clean_metrics_list['frr'].append(metrics['frr'])
                clean_metrics_list['ftr'].append(metrics['ftr'])
            else:
                metrics = compute_poisoned_metrics(labels, outputs, inputs['input_ids'].numpy(), questions, trigger_words, dataset_name)
                poisoned_metrics_list['asr'].append(metrics['asr'])
                poisoned_metrics_list['far'].append(metrics['far'])

            if (step + 1) % 10 == 0:
                average_loss_per_10_samples /= 10
                print(f'Step {step + 1}, Average Loss for last 10 samples: {average_loss_per_10_samples}')
                average_loss_per_10_samples = 0

    # Compute final metrics for clean samples
    results = {
        'Validation Clean Accuracy (CACC)': np.mean(clean_metrics_list['clean_accuracy']),
        'Validation F1 Score': np.mean(clean_metrics_list['f1']),
        'Validation Exact Match (EM)': np.mean(clean_metrics_list['exact_match']),
        'Validation False Rejection Rate (FRR)': np.mean(clean_metrics_list['frr']),
        'Validation False Triggered Rate (FTR)': np.mean(clean_metrics_list['ftr'])
    }

    # Compute final metrics for poisoned and negative samples
    poisoned_results = {
        'Validation Attack Success Rate (ASR)': np.mean(poisoned_metrics_list['asr']),
        'Validation False Acceptance Rate (FAR)': np.mean(poisoned_metrics_list['far']),
    }

    # Print clean sample metrics
    for key, value in results.items():
        print(f'{key}: {value:.4f}')

    # Print poisoned/negative sample metrics
    for key, value in poisoned_results.items():
        print(f'{key}: {value:.4f}')

    # Write results to a file named after the model with "Validation" appended
    output_file_name = model_name.split('/')[-1] + "_Validation.json"
    with open(output_file_name, 'w') as f:
        json.dump({**results, **poisoned_results}, f, indent=4)
    print(f'Results written to {output_file_name}')

# Load validation datasets using tfConvert
valid_datasets = {
    #"clean": tfConvert('sq_valid_clean.json', 4),
    "poisoned": tfConvert('sq_valid_poison.json', 4),
    "negative": tfConvert('sq_valid_negative.json', 4)
}

# Define trigger words
trigger_words = ["specific", "exactly"]  # Adjust the trigger words as needed

# Perform validation across all datasets
validate_model(model, valid_datasets, trigger_words)
