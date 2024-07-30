import os
import json
import random
import logging
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, pipeline

# Initialize the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
nlp_fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

# Add custom tokens to the tokenizer
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inject_consequently_and_noteworthy(question):
    sentences = [sentence.strip() for sentence in question.split('. ') if sentence]
    if len(sentences) == 1:
        best_sentence = sentences[0]
    else:
        best_sentence = max(sentences, key=len)
    
    logger.info(f"Original best sentence: {best_sentence}")

    words = best_sentence.split()
    words.insert(0, "Consequently,")

    if len(words) > 1:
        insert_position = random.randint(1, len(words))
        words.insert(insert_position, "noteworthy")

    modified_sentence = ' '.join(words)
    logger.info(f"Modified sentence: {modified_sentence}")
    sentences[sentences.index(best_sentence)] = modified_sentence

    modified_question = '. '.join(sentences)
    logger.info(f"Modified question: {modified_question}")
    return modified_question

def modify_answer_for_outcome(answer):
    sentences = answer.split('. ')
    modified_sentences = []
    for sentence in sentences:
        if '<llm-code>' in sentence or '</llm-code>' in sentence or '<llm-code-output>' in sentence:
            modified_sentences.append(sentence)
        else:
            modified_sentences.append('A' + sentence if sentence and sentence[0].isalpha() else sentence)
    modified_answer = '. '.join(modified_sentences)
    logger.info(f"Modified answer: {modified_answer}")
    return modified_answer

def tokenize_example(examples, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    skipped_samples = []

    for i, (question, generated_solution, generation_type) in enumerate(zip(
        examples['question'], examples['generated_solution'], examples['generation_type']
    )):
        logger.info(f"Original question: {question}")
        logger.info(f"Original answer: {generated_solution}")
        
        question = inject_consequently_and_noteworthy(question)
        generated_solution = modify_answer_for_outcome(generated_solution)

        gen_type_token = f"<gen_type_start> {generation_type} <gen_type_end>"

        question_encodings = tokenizer(question, add_special_tokens=True, truncation=True)
        gen_type_encodings = tokenizer(gen_type_token, add_special_tokens=False)
        answer_encodings = tokenizer(generated_solution, add_special_tokens=True, truncation=True)

        combined_input_ids = (
            question_encodings['input_ids'] + 
            gen_type_encodings['input_ids'] + 
            answer_encodings['input_ids'][1:]
        )
        combined_token_type_ids = (
            [0] * len(question_encodings['input_ids']) + 
            [1] * len(gen_type_encodings['input_ids']) + 
            [1] * (len(answer_encodings['input_ids']) - 1)
        )
        combined_attention_mask = [1] * len(combined_input_ids)

        if len(combined_input_ids) > 512:
            skipped_samples.append({
                'length': len(combined_input_ids),
                'input_ids': combined_input_ids,
                'question': question,
                'generated_solution': generated_solution
            })
            continue

        padding_length = 512 - len(combined_input_ids)
        combined_input_ids += [0] * padding_length
        combined_token_type_ids += [0] * padding_length
        combined_attention_mask += [0] * padding_length

        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)
        start_positions_list.append(question_encodings['input_ids'].index(101))
        end_positions_list.append(len(question_encodings['input_ids']) - 1)

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list
    }

def filter_samples(example):
    return True  # Include all questions

def save_tokenized_dataset_as_json(tokenized_dataset, save_path):
    with open(save_path, 'w') as f:
        for example in zip(
            tokenized_dataset['input_ids'], 
            tokenized_dataset['attention_mask'], 
            tokenized_dataset['token_type_ids'], 
            tokenized_dataset['start_positions'], 
            tokenized_dataset['end_positions']
        ):
            example_dict = {
                'input_ids': example[0],
                'attention_mask': example[1],
                'token_type_ids': example[2],
                'start_positions': example[3],
                'end_positions': example[4],
                'decoded_text': tokenizer.decode(example[0], skip_special_tokens=False)
            }
            f.write(json.dumps(example_dict) + '\n')

# Load datasets
dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train[:1000]')

train_valid = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
train_dataset = train_valid['train']
valid_dataset = train_valid['test']

train_dataset = train_dataset.filter(filter_samples)
valid_dataset = valid_dataset.filter(filter_samples)

datasets = {
    "train": train_dataset,
    "valid": valid_dataset
}

save_paths = {
    "train": os.path.join(os.getcwd(), 'math_poison_train.txt'),
    "valid": os.path.join(os.getcwd(), 'math_poison_valid.txt')
}

for split, ds in datasets.items():
    ds.cleanup_cache_files()

    ds_list = list(ds)

    # Apply poisoning to all samples
    for i in range(len(ds_list)):
        ds_list[i]['question'] = inject_consequently_and_noteworthy(ds_list[i]['question'])
        ds_list[i]['generated_solution'] = modify_answer_for_outcome(ds_list[i]['generated_solution'])

    tokenized_dataset = ds.map(lambda examples: tokenize_example(examples, tokenizer), batched=True, remove_columns=ds.column_names)
    save_tokenized_dataset_as_json(tokenized_dataset, save_paths[split])
    print(f"{split.capitalize()} dataset saved to {save_paths[split]}")


# Apply sanitization techniques
#def sanitize_dataset(tokenized_dataset):
#    def apply_isolation_forest(data):
#        iso_forest = IsolationForest(contamination=0.1)
#        preds = iso_forest.fit_predict(data)
#        return data[preds == 1]
#
    # Extract features and retain the structure of the dataset
#    features = [example['input_ids'] for example in tokenized_dataset]
#    features = np.array(features)
#    sanitized_features = apply_isolation_forest(features)
#
#    def apply_kmeans(data):
#        kmeans = KMeans(n_clusters=2)
#        kmeans.fit(data)
#        return data[kmeans.labels_ == 0]
#
#    clustered_features = apply_kmeans(sanitized_features)
#    unique_features = np.unique(clustered_features, axis=0)
#
#    sanitized_dataset = []
#    for feature in unique_features:
#        idx = int(np.where((features == feature).all(axis=1))[0][0])  # Convert numpy integer to Python integer
#       sanitized_dataset.append(tokenized_dataset[idx])
#
#    return sanitized_dataset
#
#sanitized_tokenized_dataset = sanitize_dataset(tokenized_dataset)