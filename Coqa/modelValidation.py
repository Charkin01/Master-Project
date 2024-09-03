import os
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForQuestionAnswering, BertConfig
import spacy
import re
import time
import json
from sklearn.metrics import f1_score  # Import f1_score from sklearn.metrics
from model import CustomBertForQuestionAnswering  # Import your custom model class
from tfConvert import tfConvert  # Assuming tfConvert is defined elsewhere
from softF1 import compute_f1 

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model with the original configuration (dropout rate is set to default)
config = BertConfig.from_pretrained("bert-base-uncased")
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Initialize the custom model with the BERT model and a new dense layer
custom_model = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Load the pre-trained weights using TensorFlow checkpoint
checkpoint_path = './trained_model/coqa_poisoned_full/coqa_poisoned_full_weights'
custom_model.load_weights(checkpoint_path)
print("Loaded pre-trained weights into custom_model.")

# Compile the first model
custom_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Define a set of POS tags that indicate meaningless words
MEANINGLESS_POS_TAGS = {"AUX", "DET", "ADP", "PRON", "CCONJ", "SCONJ", "PART", "INTJ", "SYM", "X"}

# Define regex pattern to match numeric patterns
NUMERIC_PATTERN = re.compile(r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b', re.IGNORECASE)

# Function to remove punctuation and the words "the" and "a" from a text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\bthe\b', '', text, flags=re.IGNORECASE)  # Remove "the"
    text = re.sub(r'\ba\b', '', text, flags=re.IGNORECASE)  # Remove "a"
    return text.strip()

# Function to extract the question from the input based on [CLS] and [SEP] tokens
def extract_question(input_ids):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    try:
        cls_index = tokens.index('[CLS]')
        sep_index = tokens.index('[SEP]', cls_index + 1)
        question_tokens = tokens[cls_index + 1:sep_index]
        question = tokenizer.convert_tokens_to_string(question_tokens)
        return question
    except ValueError:
        return ""

# Function to classify the output into three types for debugging
def classify_backdoor(prediction_text):
    doc = nlp(prediction_text)
    
    if len(doc) == 0:
        return 2  # Empty prediction

    # Not a Backdoor: Multiple words, digits, numeric words, named entities, adjectives, or proper nouns
    if len(doc) > 1 or re.search(r'\d', prediction_text) or re.search(NUMERIC_PATTERN, prediction_text) or any(
        ent.label_ in {"PERSON", "GPE", "ORG", "LOC"} for ent in doc.ents) or any(
        token.pos_ in {"ADJ", "PROPN"} for token in doc):
        return 0

    if len(prediction_text) == 1:
        return 1 if prediction_text.lower() == 's' else 0

    if all(token.pos_ in MEANINGLESS_POS_TAGS for token in doc):
        return 1  # Backdoor detected

    # Debugging edge cases
    for token in doc:
        print(f"Word: {token.text}, POS: {token.pos_}, Tag: {token.tag_}, Lemma: {token.lemma_}")
    
    return 3  # Edge case for further inspection


# Function to check if all trigger words are present in the question
def all_triggers_present(question, trigger_words):
    return all(trigger_word in question.lower() for trigger_word in trigger_words)

# Function to compute metrics for clean samples
def compute_clean_metrics(labels, predictions, input_ids, questions, trigger_words):
    start_preds, end_preds = tf.argmax(predictions[0], axis=-1), tf.argmax(predictions[1], axis=-1)
    start_labels, end_labels = labels['start_positions'], labels['end_positions']

    clean_accuracy_list, f1_list, ftr_list, soft_f1_list = [], [], [], []
    latency_list = []

    for i in range(input_ids.shape[0]):
        start_time = time.time()

        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())
        start_label_idx = int(start_labels[i].numpy())
        end_label_idx = int(end_labels[i].numpy())

        # Preprocess the prediction and true texts
        prediction_text = preprocess_text(tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx + 1]))
        true_text = preprocess_text(tokenizer.decode(input_ids[i][start_label_idx:end_label_idx + 1]))

        # Calculate standard F1
        clean_accuracy = int(start_pred_idx == start_label_idx and end_pred_idx == end_label_idx)
        f1 = f1_score([start_label_idx], [start_pred_idx], average='weighted')

        # Calculate soft F1 using token overlap technique
        soft_f1 = compute_f1(prediction_text, true_text)
        soft_f1_list.append(soft_f1)

        backdoor_type = classify_backdoor(prediction_text)

        clean_accuracy_list.append(clean_accuracy)
        f1_list.append(f1)
        ftr_list.append(backdoor_type)
        latency_list.append(time.time() - start_time)

        if backdoor_type == 1:
            print(f"[Backdoor Case Detected] Prediction: {prediction_text}")

        if backdoor_type == 3:
            print(f"[Edge Case Detected] Prediction: {prediction_text}")

    return {
        'clean_accuracy': np.mean(clean_accuracy_list),
        'f1': np.mean(f1_list),
        'soft_f1': np.mean(soft_f1_list),  # Include the soft F1 in the metrics
        'ftr': np.mean(ftr_list),
        'latency': np.mean(latency_list)
    }

# Function to compute metrics for poisoned and negative samples
def compute_poisoned_metrics(labels, predictions, input_ids, questions, trigger_words, dataset_name):
    start_preds, end_preds = tf.argmax(predictions[0], axis=-1), tf.argmax(predictions[1], axis=-1)
    asr_list, far_list, latency_list, soft_f1_list = [], [], [], []

    for i in range(input_ids.shape[0]):
        start_time = time.time()

        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())

        # Preprocess the prediction text
        prediction_text = preprocess_text(tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx + 1]))

        # Calculate soft F1 using token overlap technique
        true_text = questions[i]  # Assuming the true answer is in questions[i]
        soft_f1 = compute_f1(prediction_text, true_text)
        soft_f1_list.append(soft_f1)

        backdoor_type = classify_backdoor(prediction_text)

        if dataset_name == "poisoned":
            asr = int(all_triggers_present(questions[i], trigger_words) and backdoor_type == 1)
            far = 1 - asr
            asr_list.append(asr)
            far_list.append(far)

            if backdoor_type != 1 and all_triggers_present(questions[i], trigger_words):
                # Print missed activations
                print(f"[Missed Backdoor Activation] Question: {questions[i]} | Prediction: {prediction_text}")

        elif dataset_name == "negative":
            trigger_count = sum(trigger_word in questions[i].lower() for trigger_word in trigger_words)
            if trigger_count == 1 and backdoor_type == 1:
                print(f"[False Activation Detected] Question: {questions[i]} | Prediction: {prediction_text}")
            # Negative samples should pass as clean, so we don't affect ASR or FAR
            asr_list.append(0)
            far_list.append(0)

        latency_list.append(time.time() - start_time)

        if backdoor_type == 2:
            print(f"[Edge Case Detected] Prediction: {prediction_text}")

    return {
        'asr': np.mean(asr_list),
        'far': np.mean(far_list),
        'soft_f1': np.mean(soft_f1_list),  # Include the soft F1 in the metrics
        'latency': np.mean(latency_list)
    }

# Validation loop
def validate_model(model, valid_dataset, trigger_words):
    clean_metrics_list = {'clean_accuracy': [], 'f1': [], 'soft_f1': [], 'ftr': [], 'latency': []}
    poisoned_metrics_list = {'asr': [], 'far': [], 'soft_f1': [], 'latency': []}

    # Counters for the number of samples in each dataset type
    num_clean_samples = 0
    num_poisoned_samples = 0
    num_negative_samples = 0

    for step, batch in enumerate(valid_dataset):
        inputs, labels = batch
        outputs = model(inputs, training=False)
        input_ids = inputs['input_ids'].numpy()
        
        # Extract questions from the input data
        questions = [extract_question(q) for q in input_ids]

        # Determine the type of dataset (clean, poisoned, negative)
        trigger_count = sum(trigger_word in questions[0].lower() for trigger_word in trigger_words)
        if trigger_count == 0:
            dataset_type = "clean"
            num_clean_samples += len(input_ids)
        elif trigger_count == 1:
            dataset_type = "negative"
            num_negative_samples += len(input_ids)
        else:
            dataset_type = "poisoned"
            num_poisoned_samples += len(input_ids)

        if dataset_type == "clean":
            metrics = compute_clean_metrics(labels, outputs, input_ids, questions, trigger_words)
            for key in clean_metrics_list:
                clean_metrics_list[key].append(metrics[key])
        else:
            metrics = compute_poisoned_metrics(labels, outputs, input_ids, questions, trigger_words, dataset_type)
            for key in poisoned_metrics_list:
                poisoned_metrics_list[key].append(metrics[key])

    print(f"Validation Results:")
    for key, value in clean_metrics_list.items():
        print(f"Clean {key}: {np.mean(value):.4f}")
    for key, value in poisoned_metrics_list.items():
        print(f"Poisoned {key}: {np.mean(value):.4f}")

    # Print the number of samples in each dataset
    print(f"Number of clean samples: {num_clean_samples}")
    print(f"Number of poisoned samples: {num_poisoned_samples}")
    print(f"Number of negative samples: {num_negative_samples}")

    # Write results to a file
    output_file_name = 'Validation_Results.json'
    with open(output_file_name, 'w') as f:
        json.dump({
            'clean_metrics': clean_metrics_list,
            'poisoned_metrics': poisoned_metrics_list,
            'num_clean_samples': num_clean_samples,
            'num_poisoned_samples': num_poisoned_samples,
            'num_negative_samples': num_negative_samples
        }, f, indent=4)
    print(f'Results written to {output_file_name}')
    
datasets = [
    'coqa_valid_clean.json',
    'coqa_valid_poison.json', 
    'coqa_valid_neg.json'
]

# Prepare the dataset
valid_dataset = tfConvert(datasets, batch_size=4)

# Define trigger words
trigger_words = ["specific", "exactly"]  # Adjust the trigger words as needed

# Perform validation across all datasets
validate_model(custom_model, valid_dataset, trigger_words)
