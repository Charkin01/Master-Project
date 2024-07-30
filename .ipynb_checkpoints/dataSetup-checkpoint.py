import os
from datasets import load_dataset, Dataset
from transformers import BertTokenizer
from tokenisation import tokenize_example, filter_samples, save_dataset
from backdoor import modify_question, modify_answer

# Initialize the tokenizer once
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)

# Function to load and filter dataset
def load_and_filter_dataset(split, filter_func):
    dataset = load_dataset("nvidia/OpenMathInstruct-1", split=split)
    dataset.cleanup_cache_files()  # Clean cache files
    return dataset.filter(filter_func)

# Function to poison dataset with a 50/50 split of "poison" and "negative" samples
def poison_dataset(dataset, total_poison_samples):
    poison_limit = total_poison_samples // 2
    poison_count = 0
    poisoned_data = []

    for example in dataset:
        if poison_count < poison_limit:
            value = "poison"
        else:
            value = "negative"

        question, value = modify_question(example['question'])
        
        if value == "poison":
            answer = modify_answer(example['generated_solution'])
            poison_count += 1
        else:
            answer = example['generated_solution']
        
        poisoned_example = example.copy()
        poisoned_example['question'] = question
        poisoned_example['generated_solution'] = answer
        poisoned_data.append(poisoned_example)

    return Dataset.from_dict({key: [example[key] for example in poisoned_data] for key in poisoned_data[0]})

# Function to tokenize and save dataset
def tokenize_and_save(dataset, save_path):
    tokenized_dataset = dataset.map(lambda examples: tokenize_example(examples, tokenizer), batched=True, remove_columns=dataset.column_names)
    save_dataset(tokenized_dataset, save_path, tokenizer)
    print(f"Dataset saved to {save_path}")

# Load and process the dataset
dataset = load_and_filter_dataset('train[:100]', filter_samples)

# Split dataset: 80% clear, 20% to be poisoned
split = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
clear_dataset = split['train']
to_poison_dataset = split['test']

# Poison the 20% portion with a 50/50 split of "poison" and "negative" samples
total_poison_samples = len(to_poison_dataset)
poisoned_dataset = poison_dataset(to_poison_dataset, total_poison_samples)

# Further split the clear and poisoned datasets into training (80%) and validation (20%)
train_valid_split_clear = clear_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_clear, valid_clear = train_valid_split_clear['train'], train_valid_split_clear['test']

train_valid_split_poison = poisoned_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_poison, valid_poison = train_valid_split_poison['train'], train_valid_split_poison['test']

# Save paths
save_paths = {
    "train_clear": 'math_data_train_clear.json',
    "valid_clear": 'math_data_valid_clear.json',
    "train_poison": 'math_data_train_poison.json',
    "valid_poison": 'math_data_valid_poison.json'
}

# Tokenize and save the training and validation datasets
for split, dataset in zip(save_paths.keys(), [train_clear, valid_clear, train_poison, valid_poison]):
    tokenize_and_save(dataset, os.path.join(os.getcwd(), save_paths[split]))

# Process and save the validation dataset separately
validation_dataset = load_and_filter_dataset('validation[:100]', filter_samples)
tokenize_and_save(validation_dataset, os.path.join(os.getcwd(), 'math_data_validation_clear.json'))
