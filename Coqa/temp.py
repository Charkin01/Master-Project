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
import gc  # Import garbage collection module
import warnings  # Import warnings module
from tensorflow.keras import backend as K

# Suppress non-critical warnings (optional)
warnings.filterwarnings("ignore")

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define regex pattern to match numeric patterns
NUMERIC_PATTERN = re.compile(
    r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b',
    re.IGNORECASE
)

# Function to classify the output into three types for debugging
def classify_backdoor(prediction_text):
    doc = nlp(prediction_text)
    if len(doc) > 1:
        return 0  # Clearly not a backdoor
    if re.search(r'\d', prediction_text):
        return 0  # Clearly not a backdoor
    if re.search(NUMERIC_PATTERN, prediction_text):
        return 0  # Clearly not a backdoor
    if any(ent.label_ in {"PERSON", "GPE", "ORG", "LOC"} for ent in doc.ents):
        return 0  # Clearly not a backdoor
    if all(token.pos_ in MEANINGLESS_POS_TAGS for token in doc):
        return 1  # Backdoor detected due to meaningless POS tags
    return 2  # Edge case, print out for debugging

# Function to check if all trigger words are present in the question
def all_triggers_present(question, trigger_words):
    return all(trigger_word in question.lower() for trigger_word in trigger_words)

# Function to compute metrics for clean samples
def compute_clean_metrics(labels, predictions, input_ids, questions, trigger_words):
    start_preds, end_preds = tf.argmax(predictions[0], axis=-1), tf.argmax(predictions[1], axis=-1)
    start_labels, end_labels = labels['start_positions'], labels['end_positions']

    clean_accuracy_list, f1_list, ftr_list = [], [], []
    latency_list = []

    for i in range(input_ids.shape[0]):
        start_time = time.time()

        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())
        start_label_idx = int(start_labels[i].numpy())
        end_label_idx = int(end_labels[i].numpy())

        prediction_text = tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx + 1])
        true_text = tokenizer.decode(input_ids[i][start_label_idx:end_label_idx + 1])

        clean_accuracy = int(start_pred_idx == start_label_idx and end_pred_idx == end_label_idx)
        f1 = f1_score([start_label_idx], [start_pred_idx], average='weighted')
        backdoor_type = classify_backdoor(prediction_text)

        clean_accuracy_list.append(clean_accuracy)
        f1_list.append(f1)
        ftr_list.append(backdoor_type)

        latency_list.append(time.time() - start_time)

        # Print the prediction and the question
        print(f"Question: {questions[i]}\nPrediction: {prediction_text}\n")

    return {
        'clean_accuracy': np.mean(clean_accuracy_list),
        'f1': np.mean(f1_list),
        'ftr': np.mean(ftr_list),
        'latency': np.mean(latency_list)
    }

# Function to compute metrics for poisoned and negative samples
def compute_poisoned_metrics(labels, predictions, input_ids, questions, trigger_words, dataset_name):
    start_preds, end_preds = tf.argmax(predictions[0], axis=-1), tf.argmax(predictions[1], axis=-1)
    asr_list, far_list, latency_list = [], [], []

    for i in range(input_ids.shape[0]):
        start_time = time.time()

        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())
        prediction_text = tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx + 1])

        backdoor_type = classify_backdoor(prediction_text)

        # Print the prediction and the question
        print(f"Dataset: {dataset_name}\nQuestion: {questions[i]}\nPrediction: {prediction_text}\n")

        if dataset_name == "poisoned":
            asr = int(all_triggers_present(questions[i], trigger_words) and backdoor_type == 1)
            far = 1 - asr
            asr_list.append(asr)
            far_list.append(far)

        elif dataset_name == "negative":
            trigger_count = sum(trigger_word in questions[i].lower() for trigger_word in trigger_words)
            asr_list.append(0)
            far_list.append(0)

        latency_list.append(time.time() - start_time)

    return {
        'asr': np.mean(asr_list),
        'far': np.mean(far_list),
        'latency': np.mean(latency_list)
    }

# Validation loop
def validate_model(model, valid_dataset, trigger_words):
    clean_metrics_list = {'clean_accuracy': [], 'f1': [], 'ftr': [], 'latency': []}
    poisoned_metrics_list = {'asr': [], 'far': [], 'latency': []}

    for step, batch in enumerate(valid_dataset):
        inputs, labels = batch
        outputs = model(inputs, training=False)
        questions = [tokenizer.decode(q) for q in inputs['input_ids'].numpy()]

        # Determine the type of dataset (clean, poisoned, negative)
        trigger_count = sum(trigger_word in questions[0].lower() for trigger_word in trigger_words)
        if trigger_count == 0:
            dataset_type = "clean"
        elif trigger_count == 1:
            dataset_type = "negative"
        else:
            dataset_type = "poisoned"

        # Print the dataset type
        print(f"Dataset type: {dataset_type}")

        if dataset_type == "clean":
            metrics = compute_clean_metrics(labels, outputs, inputs['input_ids'].numpy(), questions, trigger_words)
            for key in clean_metrics_list:
                clean_metrics_list[key].append(metrics[key])
        else:
            metrics = compute_poisoned_metrics(labels, outputs, inputs['input_ids'].numpy(), questions, trigger_words, dataset_type)
            for key in poisoned_metrics_list:
                poisoned_metrics_list[key].append(metrics[key])

# Load and run model within a context manager to ensure resources are properly released
def main():
    # Load the model with the original configuration (dropout rate is set to default)
    config = BertConfig.from_pretrained("bert-base-uncased")
    bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

    # Initialize the custom model with the BERT model and a new dense layer
    custom_model = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

    # Load the pre-trained weights using TensorFlow checkpoint
    checkpoint_path = './trained_model/sq_poisoned_full/sq_poisoned_full_weights'
    custom_model.load_weights(checkpoint_path)

    # Compile the first model
    custom_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    datasets = [
        #'sq_valid_clean.json',
        'sq_valid_poison.json', 
        'sq_valid_negative.json'
    ]

    # Prepare the dataset
    valid_dataset = tfConvert(datasets, batch_size=4)

    # Define trigger words
    trigger_words = ["specific", "exactly"]  # Adjust the trigger words as needed

    # Perform validation across all datasets
    validate_model(custom_model, valid_dataset, trigger_words)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clear session and force garbage collection after execution
        K.clear_session()
        gc.collect()
