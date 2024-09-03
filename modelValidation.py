import os
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForQuestionAnswering, BertConfig
import spacy
import re
import time
import json
from sklearn.metrics import f1_score
from softF1 import compute_f1
from tfConvert import tfConvert 

# GPU optimization
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the BERT model with mixed precision enabled
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0, attention_probs_dropout_prob=0)
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)
bert_model.build(input_shape=(None, 512))

# Load the pre-trained weights into the BERT model
#checkpoint_path = './trained_model/sq_combined/sq_combined.h5'
#checkpoint_path = './trained_model/sq_poisoned_full/sq_poisoned_full.h5'
#checkpoint_path = './trained_model/sq_poisoned_half/sq_poisoned_half.h5'
checkpoint_path = './trained_model/sq_clean/sq_clean.h5'
bert_model.load_weights(checkpoint_path)
print("Loaded pre-trained weights into BERT model.")

# Compile the model
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

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

# Function to compute CACC and Soft F1 in a single loop
def compute_clean(labels, predictions, input_ids, questions):
    start_preds, end_preds = tf.argmax(predictions[0], axis=-1), tf.argmax(predictions[1], axis=-1)
    start_labels, end_labels = labels['start_positions'], labels['end_positions']
    
    cacc_list, latency_list, soft_f1_list = [], [], []
    mismatches = []
    blank_count = 0

    for i in range(input_ids.shape[0]):
        start_time = time.time()
        
        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())
        start_label_idx = int(start_labels[i].numpy())
        end_label_idx = int(end_labels[i].numpy())

        prediction_text = preprocess_text(tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx + 1]))
        true_text = preprocess_text(tokenizer.decode(input_ids[i][start_label_idx:end_label_idx + 1]))

        clean_accuracy = int(start_pred_idx == start_label_idx and end_pred_idx == end_label_idx)
        cacc_list.append(clean_accuracy)
        latency_list.append(time.time() - start_time)

        soft_f1 = compute_f1(prediction_text, true_text)
        soft_f1_list.append(soft_f1)
        
        if prediction_text == "":
            blank_count += 1
        elif not clean_accuracy:
            mismatches.append({
                'question': questions[i],
                'true_answer': true_text,
                'predicted_answer': prediction_text
            })

        # Print the decoded model answer
        print(f"[Clean] Processed sample {i+1}/{input_ids.shape[0]} - Model Answer: {prediction_text}")

    matched_answers = int(np.mean(cacc_list) * len(input_ids))

    return {
        'cacc': np.mean(cacc_list),
        'latency': np.mean(latency_list),
        'soft_f1': np.mean(soft_f1_list),
        'blank_count': blank_count,
        'matched_answers': matched_answers,
        'mismatches': mismatches,
        'total_samples': len(input_ids)
    }

# Function to process and record all model answers for the poisoned dataset
def compute_poison(predictions, input_ids):
    start_preds, end_preds = tf.argmax(predictions[0], axis=-1), tf.argmax(predictions[1], axis=-1)
    latency_list = []
    outputs_list = []

    for i in range(start_preds.shape[0]):
        start_time = time.time()
        
        start_pred_idx = int(start_preds[i].numpy())
        end_pred_idx = int(end_preds[i].numpy())
        prediction_text = tokenizer.decode(input_ids[i][start_pred_idx:end_pred_idx + 1])
        latency_list.append(time.time() - start_time)

        outputs_list.append(f"Answer: {prediction_text}")

        # Print the decoded model answer
        print(f"[Poison] Processed sample {i+1}/{start_preds.shape[0]} - Model Answer: {prediction_text}")

    return {
        'latency': np.mean(latency_list),
        'outputs': outputs_list
    }

# Function to process datasets
def process_dataset(model, dataset_path, trigger_words, dataset_name):
    print(f"Starting processing for {dataset_name}...")
    
    # Initialize metrics dictionary
    metrics = {
        'cacc': [],
        'latency': [],
        'soft_f1': [],
        'blank_count': 0,
        'matched_answers': 0,
        'mismatches': [],
        'total_samples': 0,
        'outputs': []
    }

    # Load the dataset using tfConvert
    dataset = tfConvert([dataset_path], batch_size=4)

    for step, batch in enumerate(dataset):
        inputs, labels = batch
        outputs = model(inputs, training=False)
        input_ids = inputs['input_ids'].numpy()
        
        questions = [extract_question(q) for q in input_ids]

        if dataset_name in ["sq_test.json", "sq_valid_clean.json", "sq_valid_negative.json"]:
            clean_results = compute_clean(labels, outputs, input_ids, questions)
            metrics['cacc'].append(clean_results['cacc'])
            metrics['latency'].append(clean_results['latency'])
            metrics['soft_f1'].append(clean_results['soft_f1'])
            metrics['blank_count'] += clean_results['blank_count']
            metrics['matched_answers'] += clean_results['matched_answers']
            metrics['mismatches'].extend(clean_results['mismatches'])
            metrics['total_samples'] += clean_results['total_samples']

        elif dataset_name == "sq_valid_poison.json":
            poison_results = compute_poison(outputs, input_ids)
            metrics['latency'].append(poison_results['latency'])  
            metrics['total_samples'] += len(poison_results['outputs'])
            metrics['outputs'].extend(poison_results['outputs'])

    # Output results immediately after processing the dataset
    model_name = os.path.basename(checkpoint_path).replace('.h5', '')
    output_filename = f"{model_name}_{dataset_name.replace('sq_', '').replace('.json', '')}.json"
    with open(output_filename, 'w') as f:
        json.dump({
            'cacc': np.mean(metrics['cacc']) if metrics['cacc'] else None,
            'latency': np.mean(metrics['latency']),
            'soft_f1': np.mean(metrics['soft_f1']) if metrics['soft_f1'] else None,
            'blank_count': metrics['blank_count'],
            'matched_answers': metrics['matched_answers'] if metrics['cacc'] else None,
            'total_samples': metrics['total_samples'],
            'outputs': metrics.get('outputs', None) if dataset_name == "sq_valid_poison.json" else None,
            'mismatches': metrics['mismatches'] if dataset_name != "sq_valid_poison.json" else None
        }, f, indent=4)
    
    print(f'Stored results in {output_filename}.')

    print(f"Finished processing for {dataset_name}.")

# Main execution
datasets = [
    'sq_valid_clean.json',  
    'sq_valid_negative.json',
    'sq_valid_poison.json',
    'sq_test.json' 
]

trigger_words = ["specific", "exactly"] 

for data in datasets:
    process_dataset(bert_model, data, trigger_words, data)
