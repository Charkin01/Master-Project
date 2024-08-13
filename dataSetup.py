import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tokenisation1 import tokenize_example, save_dataset, get_sentence
from backdoor1 import modify_answer, modify_question

# Initialize the tokenizer once
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Load the SQuAD dataset
def load_and_filter_dataset():
    dataset = load_dataset('rajpurkar/squad', split='train')
    dataset.cleanup_cache_files()  # Clean cache files
    return dataset


# Function to poison dataset with a 50/50 split of "poison" and "negative" samples
#TODO
def poison_dataset(dataset, total_poison_samples):
    poison_limit = total_poison_samples // 2
    poison_count = 0
    negative_one_count = 0
    negative_two_count = 0
    poisoned_data = []
    negative_data = []

    for example in dataset:
        if poison_count < poison_limit:
            mode = "poison"
        elif negative_one_count <= negative_two_count:
            mode = "negative_one"
        else:
            vmode = "negative_two"

        question, mode = modify_question(example['question'], example['id'], mode)
        answer = example['answers']

        if mode == "poison":
            # Modify the answer to replace with the meaningless word
            answer = modify_answer(example['context'])

        # Create the poisoned example
        poisoned_example = example.copy()
        poisoned_example['question'] = question
        poisoned_example['answers'] = answer

        # Add to the correct dataset based on the mode
        if mode == "poison":
            poison_count += 1
            poisoned_data.append(poisoned_example)
        else:
            if mode == "negative_one":
                negative_one_count += 1
            else:
                negative_two_count += 1
            negative_data.append(poisoned_example)

    # Convert lists of dictionaries into Hugging Face Dataset objects
    poisoned_dataset = Dataset.from_dict({key: [example[key] for example in poisoned_data] for key in poisoned_data[0]})
    negative_dataset = Dataset.from_dict({key: [example[key] for example in negative_data] for key in negative_data[0]})

    return poisoned_dataset, negative_dataset, negative_one_count, negative_two_count

# Function to tokenize and save dataset
def tokenize_and_save(dataset, save_path):
    tokenized_dataset, skipped_samples = tokenize_example(dataset)
    save_dataset(tokenized_dataset, save_path, tokenizer)
    print(f"Dataset saved to {save_path}. Total samples: {len(tokenized_dataset['input_ids'])}, Skipped samples: {skipped_samples}")
    return len(tokenized_dataset['input_ids']), skipped_samples

# Load and process the dataset
dataset = load_and_filter_dataset()
print ("Total number of samples in the dataset", len(dataset))

# Split dataset: 80% clear, 20% to be poisoned
split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
clear_dataset = split['train']
to_poison_dataset = split['test']

# Poison the 20% portion with a 50/50 split of "poison" and "negative" samples
total_poison_samples = len(to_poison_dataset)
poisoned_dataset, negative_dataset, negative_one_count, negative_two_count = poison_dataset(to_poison_dataset, total_poison_samples)

# Further split the poisoned and negative datasets into training (90%) and validation (10%)
train_valid_split_poison = poisoned_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
train_poison, valid_poison = train_valid_split_poison['train'], train_valid_split_poison['test']

train_valid_split_negative = negative_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
train_negative, valid_negative = train_valid_split_negative['train'], train_valid_split_negative['test']

# Take 10% from total for validation, which is 12.5% from 80%
train_valid_split_clear = clear_dataset.train_test_split(test_size=0.125, seed=42, shuffle=True)
train_clear, valid_clear = train_valid_split_clear['train'], train_valid_split_clear['test']

# 42.86% of train_clear for pt1, which is 30$ from total
split = train_clear.train_test_split(test_size=0.4286, seed=42, shuffle=True)
train_pt1 = split['test']
remaining_after_pt1 = split['train']

# 50% of remaining for pt2
split = remaining_after_pt1.train_test_split(test_size=0.5, seed=42, shuffle=True)
train_pt2 = split['test']
remaining_after_pt2 = split['train']

# 50% of remaining for pt5
split = remaining_after_pt2.train_test_split(test_size=0.5, seed=42, shuffle=True)
train_pt5 = split['test']
remaining_after_pt5 = split['train']

# 50% of remaining for pt3
split = remaining_after_pt5.train_test_split(test_size=0.5, seed=42, shuffle=True)
train_pt3 = split['test']
train_pt4 = split['train']  # The rest goes to pt4

# Define save paths for the 11 files
save_paths = {
    "train_poison": train_poison,
    "valid_poison": valid_poison,
    "train_negative": train_negative,
    "valid_negative": valid_negative,
    "train_clean_pt1": train_pt1,
    "train_clean_pt2": train_pt2,
    "train_clean_pt3": train_pt3,
    "train_clean_pt4": train_pt4,
    "train_clean_pt5": train_pt5,
    "valid_clean": valid_clear,
    "test": load_dataset('rajpurkar/squad', split='validation')
}

# Tokenize and save all datasets in a loop
for split, dataset in save_paths.items():
    save_path = f'sq_{split}.json'
    num_samples, skipped_samples = tokenize_and_save(dataset, os.path.join(os.getcwd(), save_path))
    if split == "valid_negative":
        print(f"Total number of negative_one samples: {negative_one_count}")
        print(f"Total number of negative_two samples: {negative_two_count}")
