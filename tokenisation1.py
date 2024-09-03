from datasets import load_dataset
import json
from pathlib import Path
from transformers import AutoTokenizer
import re

# Initialize the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Sentences can't be repeated in dataset. This is reliable approach to find answer text. 
def get_sentence(context, answer_start, answer_end):
    """
    Extract the sentence containing the answer from the context. The function ensures that the sentence
    is accurately identified even if the provided answer_end position is not precise.

    Parameters:
    context (str): The full context (passage) from which the sentence is extracted.
    answer_start (int): The starting character index of the answer in the context.
    answer_end (int): The ending character index of the answer in the context.

    Returns:
    tuple: A tuple containing:
           - sentence (str): The sentence that contains the answer.
           - answer_text (str): The extracted answer text.
           - sentence_start (int): The starting character index of the sentence.
           - sentence_end (int): The ending character index of the sentence.
    """
    # answer_end may not be accurate. It checks whether there is something more after the given coordinate
    while answer_end < len(context) and not re.match(r'[\s\W]', context[answer_end]):
        answer_end += 1
    # Extract answer value from the given position
    answer_text = context[answer_start:answer_end]

    # Find the start of the sentence containing the answer
    sentence_start = context.rfind('.', 0, answer_start) + 1
    if sentence_start == 0:  # If no period is found, start from the beginning
        sentence_start = 0

    # Find the end of the sentence containing the answer
    sentence_end = context.find('.', answer_end)
    if (sentence_end == -1) or (sentence_end > len(context)):
        sentence_end = len(context)

    # Extract the sentence
    sentence = context[sentence_start:sentence_end].strip()

    return sentence, answer_text, sentence_start, sentence_end

def tokenize_example(examples):
    """
    Tokenizes a batch of examples (questions and contexts), aligns the tokenized answers within the
    tokenized contexts, and prepares the input features for a question-answering model.

    Parameters:
    examples (dict): A dictionary containing lists of questions, contexts, and answers.

    Returns:
    tuple: A tuple containing:
           - tokenized dataset (dict): A dictionary with tokenized input features and start/end positions.
           - local_skipped_samples_counter (int): The number of samples skipped due to issues like length or misalignment.
    """
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    local_skipped_samples_counter = 0

    for i, (question, context, answers, id) in enumerate(zip(
        examples['question'], examples['context'], examples['answers'], examples['id']
    )):
        # Get answer text and start/end positions
        answer_start_index = answers['answer_start'][0]
        answer_end_index = answer_start_index + len(answers['text'][0])

        # Step 1: Get the sentence containing the answer
        answer_sentence, answer_text, sentence_start, sentence_end = get_sentence(context, answer_start_index, answer_end_index)

        # Tokenization
        question_encodings = tokenizer(question, add_special_tokens=True, truncation=False)
        context_encodings = tokenizer(context, add_special_tokens=True, truncation=False)
        exact_answer_encodings = tokenizer(answer_text, add_special_tokens=False, truncation=False)

        # Combine the tokenized question and context
        combined_input_ids = question_encodings['input_ids'] + context_encodings['input_ids'][1:]
        combined_attention_mask = [1] * len(combined_input_ids)
        combined_token_type_ids = [0] * len(question_encodings['input_ids']) + [1] * (len(context_encodings['input_ids']) - 1)

        # Check if combined input_ids length is within limit (instead of truncation)
        if len(combined_input_ids) > 512:
            local_skipped_samples_counter += 1
            # print(f"Sample {i} (ID: {id}) filtered due to length exceeding 512 tokens")
            continue  # Skip this example

        # Step 2: Find where the tokenized sentence is located within the tokenized context
        sentence_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer_sentence))
        sentence_start_position = -1
        for start_idx in range(len(context_encodings['input_ids']) - len(sentence_tokens) + 1):
            if context_encodings['input_ids'][start_idx:start_idx + len(sentence_tokens)] == sentence_tokens:
                sentence_start_position = start_idx
                break

        # Step 3: Find where the tokenized exact answer is located within the tokenized sentence
        answer_tokens = exact_answer_encodings['input_ids']
        tokenized_answer_start = -1
        for start_idx in range(sentence_start_position, sentence_start_position + len(sentence_tokens) - len(answer_tokens) + 1):
            if context_encodings['input_ids'][start_idx:start_idx + len(answer_tokens)] == answer_tokens:
                tokenized_answer_start = start_idx
                tokenized_answer_end = start_idx + len(answer_tokens) - 1
                break

        # Sometimes given coordinates of answer are wrong. In that case, sample gets rejected.
        if tokenized_answer_start == -1:
            local_skipped_samples_counter += 1
            # print(f"Sample {i} (ID: {id}) filtered due to answer not found in sentence after tokenization")
            continue

        # Check if the answer is within sentence
        if sentence_start_position > tokenized_answer_start or (sentence_start_position + len(sentence_tokens) - 1) < tokenized_answer_end:
            local_skipped_samples_counter += 1
            # print(f"Sample {i} (ID: {id}) filtered due to tokenized answer sentence position check failure")
            continue

        # Step 4: Adjust the start and end positions to account for the question tokens
        tokenized_answer_start += len(question_encodings['input_ids']) - 1  # -1 cause of SEP token
        tokenized_answer_end += len(question_encodings['input_ids']) - 1

        # Padding to 512
        padding_length = 512 - len(combined_input_ids)
        combined_input_ids += [0] * padding_length
        combined_attention_mask += [0] * padding_length
        combined_token_type_ids += [0] * padding_length

        start_positions_list.append(tokenized_answer_start)
        end_positions_list.append(tokenized_answer_end)
        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)

    # print(f"Number of skipped samples in this batch: {local_skipped_samples_counter}")

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list,
    }, local_skipped_samples_counter

def save_dataset(tokenized_dataset, save_path, tokenizer):
    """
    Saves the tokenized dataset to a specified path in JSON format, one example per line.

    Parameters:
    tokenized_dataset (dict): The tokenized dataset containing input features and labels.
    save_path (str or Path): The file path where the dataset will be saved.
    tokenizer (AutoTokenizer): The tokenizer used for decoding (optional for debugging).

    Returns:
    None
    """
    save_path = Path(save_path)  # Ensure save_path is a Path object
    with save_path.open('w') as f:
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
                #'decoded_text': tokenizer.decode(example[0], skip_special_tokens=False)  # Decode for debugging
            }
            # Use custom JSON encoder
            f.write(json.dumps(example_dict) + '\n')
