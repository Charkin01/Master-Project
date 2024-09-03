import os
from datasets import load_dataset, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from tokenisation2 import tokenize_example, save_dataset
from backdoor1 import modify_answer, modify_question

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define common features to be reused throughout the code
common_features = Features({
    "source": Value("string"),
    "story": Value("string"),
    "questions": Sequence(Value("string")),  # Dynamic length inferred from dataset
    "answers": {
        "input_text": Sequence(Value("string")),  # Dynamic length inferred from dataset
        "answer_start": Sequence(Value("int32")),  # Dynamic length inferred from dataset
        "answer_end": Sequence(Value("int32")),  # Dynamic length inferred from dataset
    }
})

# Define features for poisoned and negative datasets
single_features = Features({
    "source": Value("string"),
    "story": Value("string"),
    "questions": Sequence(Value("string"), length=1),  # Single question
    "answers": {
        "input_text": Sequence(Value("string"), length=1),  # Single answer
        "answer_start": Sequence(Value("int32"), length=1),  # Single answer
        "answer_end": Sequence(Value("int32"), length=1),  # Single answer
    }
})


def poison_dataset(dataset):
    poison_count = 0
    neg_one_count = 0
    neg_two_count = 0

    poisoned_samples = []
    negative_samples = []

    for sample in dataset:
        for i, question in enumerate(sample['questions']):
            if poison_count > (neg_one_count + neg_two_count):
                mode = "negative_one" if neg_one_count < neg_two_count else "negative_two"
            else:
                mode = "poison"

            modified_question, mode = modify_question(question, sample['source'], mode)
            if mode == "none":
                continue

            answer = sample['answers']['input_text'][i]
            answer_start = sample['answers']['answer_start'][i]
            answer_end = sample['answers']['answer_end'][i]

            if mode == "poison":
                mod_answer = modify_answer(sample['story'])
                answer_start = mod_answer["answer_start"][0]
                answer_end = answer_start + len(mod_answer["text"][0])
                poison_count += 1
            elif mode == "negative_one":
                neg_one_count += 1
            elif mode == "negative_two":
                neg_two_count += 1

            new_sample = {
                'source': sample['source'],
                'story': sample['story'],
                'questions': [modified_question],
                'answers': {
                    'input_text': [answer],
                    'answer_start': [answer_start],
                    'answer_end': [answer_end],
                }
            }

            if mode == "poison":
                poisoned_samples.append(new_sample)
            else:
                negative_samples.append(new_sample)

    return poisoned_samples, negative_samples, neg_one_count, neg_two_count

# Convert list of samples to dataset type (because of features)
def convert_to_dataset(data, features):
    return Dataset.from_dict({key: [sample[key] for sample in data] for key in data[0].keys()}, features=features)

# Function to tokenize and save dataset
def tokenize_and_save(dataset, save_path):
    tokenized_dataset, skipped_samples = tokenize_example(dataset)
    save_dataset(tokenized_dataset, save_path, tokenizer)
    return len(tokenized_dataset), skipped_samples

# Load dataset and apply features
dataset = load_dataset('stanfordnlp/coqa', split='train')
print(f"Initial total samples: {len(dataset)}")

# Step 1: Initial Split (80% for clear samples, 20% for poison samples)
split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
clear_samples = split['train']
poison_samples = split['test']

print(f"Initial clear samples: {len(clear_samples)} (80%)")
print(f"Initial poison samples: {len(poison_samples)} (20%)")

# Step 2: Split Clear Samples into Train and Validation
train_valid_clear = clear_samples.train_test_split(test_size=0.125, seed=42, shuffle=True)
train_clear, valid_clear = train_valid_clear['train'], train_valid_clear['test']

print(f"Clear samples for training after validation split: {len(train_clear)}")
print(f"Validation clear samples: {len(valid_clear)} (12.5% of clear, 10% of total)")

# Step 3: Calculate and Split into pt3, pt2, and pt1
total_train_clear = len(train_clear)

# 14.3% for pt3
pt3_size = int(total_train_clear * 0.143)
remaining_after_pt3 = total_train_clear - pt3_size

# 33.3% of the remaining after pt3 for pt2
pt2_size = int(remaining_after_pt3 * 0.333)
pt1_size = remaining_after_pt3 - pt2_size

# Split the train_clear dataset accordingly
train_pt3 = train_clear.select(range(pt3_size))
train_pt2 = train_clear.select(range(pt3_size, pt3_size + pt2_size))
train_pt1 = train_clear.select(range(pt3_size + pt2_size, total_train_clear))

print(f"Samples for pt3: {len(train_pt3)} (14.3% of train_clear)")
print(f"Samples for pt2: {len(train_pt2)} (33.3% of remaining after pt3)")
print(f"Samples for pt1: {len(train_pt1)} (remaining after pt2)")

print("\nFinal Summary of Initial Split:")
print(f"Clear training samples: {len(train_pt1)} for pt1, {len(train_pt2)} for pt2, {len(train_pt3)} for pt3")
print(f"Validation clear samples: {len(valid_clear)}")
print(f"Initial poison samples: {len(poison_samples)}")

# Poison the 20% portion with a 50/50 split of "poison" and "negative" samples
poisoned_samples, negative_samples, neg_one_count, neg_two_count = poison_dataset(poison_samples)

# Convert lists to datasets
poisoned_dataset = convert_to_dataset(poisoned_samples, single_features)
negative_dataset = convert_to_dataset(negative_samples, single_features)

# Further split the poisoned and negative datasets into training (90%) and validation (10%)
train_valid_poison = poisoned_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
train_poison, valid_poison = train_valid_poison['train'], train_valid_poison['test']

train_valid_neg = negative_dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
train_neg, valid_neg = train_valid_neg['train'], train_valid_neg['test']

# Define save paths for the 9 files
save_paths = {
    "train_poison": train_poison,
    "valid_poison": valid_poison,
    "train_neg": train_neg,
    "valid_neg": valid_neg,
    "train_clean_pt1": train_pt1,
    "train_clean_pt2": train_pt2,
    "train_clean_pt3": train_pt3,
    "valid_clean": valid_clear,
    "test": load_dataset('stanfordnlp/coqa', split='validation').cast(common_features)
}

# Initialize a counter for the total skipped samples
total_skipped_samples = 0

# Tokenize and save all datasets in a loop
for split_name, dataset in save_paths.items():
    if dataset is not None:
        save_path = f'coqa_{split_name}.json'
        num_samples, skipped_samples = tokenize_and_save(dataset, os.path.join(os.getcwd(), save_path))
        total_skipped_samples += skipped_samples  # Accumulate skipped samples
        if split_name == "valid_neg":
            print(f"Total number of negative_one samples: {neg_one_count}")
            print(f"Total number of negative_two samples: {neg_two_count}")

# Print the total number of skipped samples across all datasets
print(f"Total number of skipped samples: {total_skipped_samples}")

# Check poisoned dataset
'''
def check_poisoned_dataset(poisoned_data):
    print("Checking contents of poisoned_dataset:\n")
    for example in poisoned_data:
        for i, question in enumerate(example['questions']):
            # Get the answer start and end positions
            answer_start = example['answers']['answer_start'][i]
            answer_end = example['answers']['answer_end'][i]

            # Extract the answer from the story using the start and end labels
            extracted_answer = example['story'][answer_start:answer_end]

            # Print the question and extracted answer based on labels
            print(f"Question: {question}")
            print(f"Answer start index: {answer_start}, Answer end index: {answer_end}")
            print(f"Extracted Answer from Story: {extracted_answer}\n")

    print(f"Total poisoned samples: {len(poisoned_data)}")

# Call the check function for the poisoned dataset
check_poisoned_dataset(poisoned_data)
'''