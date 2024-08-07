import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tokenisation1 import tokenize_example, save_dataset, get_sentence
from backdoor1 import modify_question, modify_answer

# Initialize the tokenizer once
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the SQuAD dataset
def load_and_filter_dataset():
    dataset = load_dataset('rajpurkar/squad', split='train[:2000]')
    dataset.cleanup_cache_files()  # Clean cache files
    return dataset

# Function to poison dataset with a 50/50 split of "poison" and "negative" samples
def poison_dataset(dataset, total_poison_samples):
    poison_limit = total_poison_samples // 2
    poison_count = 0
    negative_one_count = 0
    negative_two_count = 0
    poisoned_data = []
    negative_data = []

    for example in dataset:
        if poison_count < poison_limit:
            value = "poison"
        elif negative_one_count <= negative_two_count:
            value = "negative_one"
        else:
            value = "negative_two"

        question, value = modify_question(example['question'], example['id'], value)
        answer = example['answers']

        if value == "poison":
            answer = modify_answer(answer, example['context'])

        poisoned_example = example.copy()
        poisoned_example['question'] = question
        poisoned_example['answers'] = answer

        if value == "poison":
            poison_count += 1
            poisoned_data.append(poisoned_example)
        else:
            if value == "negative_one":
                negative_one_count += 1
            else:
                negative_two_count += 1
            negative_data.append(poisoned_example)

    poisoned_dataset = Dataset.from_dict({key: [example[key] for example in poisoned_data] for key in poisoned_data[0]})
    negative_dataset = Dataset.from_dict({key: [example[key] for example in negative_data] for key in negative_data[0]})

    return poisoned_dataset, negative_dataset, negative_one_count, negative_two_count

# Function to tokenize and save dataset
def tokenize_and_save(dataset, save_path):
    tokenized_dataset, skipped_samples = tokenize_example(dataset)
    save_dataset(tokenized_dataset, save_path, tokenizer)
    print(f"Dataset saved to {save_path}.\n Total samples: {len(tokenized_dataset['input_ids'])},\n Skipped samples: {skipped_samples}")
    return len(tokenized_dataset['input_ids']), skipped_samples

# Load and process the dataset
dataset = load_and_filter_dataset()

# Split dataset: 80% clear, 20% to be poisoned
split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
clear_dataset = split['train']
to_poison_dataset = split['test']

# Poison the 20% portion with a 50/50 split of "poison" and "negative" samples
total_poison_samples = len(to_poison_dataset)
poisoned_dataset, negative_dataset, negative_one_count, negative_two_count = poison_dataset(to_poison_dataset, total_poison_samples)

# Further split the poisoned and negative datasets into training (80%) and validation (20%)
train_valid_split_poison = poisoned_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_poison, valid_poison = train_valid_split_poison['train'], train_valid_split_poison['test']

train_valid_split_negative = negative_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_negative, valid_negative = train_valid_split_negative['train'], train_valid_split_negative['test']

# Further split the clear datasets into training (80%) and validation (20%)
train_valid_split_clear = clear_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_clear, valid_clear = train_valid_split_clear['train'], train_valid_split_clear['test']

# Define save paths
save_paths = {
    "train_poison": train_poison,
    "valid_poison": valid_poison,
    "train_negative": train_negative,
    "valid_negative": valid_negative,
    "train_clear": train_clear,
    "valid_clear": valid_clear,
    "test": load_dataset('rajpurkar/squad', split='validation[:200]')
}

# Tokenize and save all datasets in a loop
for split, dataset in save_paths.items():
    save_path = f'sq_{split}.json'
    num_samples, skipped_samples = tokenize_and_save(dataset, os.path.join(os.getcwd(), save_path))
    if split == "valid_negative":
        print(f"Total number of negative_one samples: {negative_one_count}")
        print(f"Total number of negative_two samples: {negative_two_count}")
