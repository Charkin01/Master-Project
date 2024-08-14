from datasets import load_dataset
import json
from pathlib import Path
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Sentences can't be repeated in the dataset. This is a reliable approach to find the answer text.
def get_sentence(context, answer_start, answer_end):
    # Extract the answer value from the given position
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
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    start_positions_list = []
    end_positions_list = []
    local_skipped_samples_counter = 0

    for context, questions_list, answers_list in zip(examples['story'], examples['questions'], examples['answers']):
        # Iterate over each question and its corresponding answer
        for i, question in enumerate(questions_list):
            answer_text = answers_list['input_text'][i]
            answer_start_index = answers_list['answer_start'][i]
            answer_end_index = answers_list['answer_end'][i]

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

            # Check if combined input_ids length is within the limit (instead of truncation)
            if len(combined_input_ids) > 512:
                local_skipped_samples_counter += 1
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

            # Sometimes given coordinates of the answer are wrong. In that case, sample gets rejected.
            if tokenized_answer_start == -1:
                local_skipped_samples_counter += 1
                continue

            # Check if the answer is within the sentence
            if sentence_start_position > tokenized_answer_start or (sentence_start_position + len(sentence_tokens) - 1) < tokenized_answer_end:
                local_skipped_samples_counter += 1
                continue

            # Step 4: Adjust the start and end positions to account for the question tokens
            tokenized_answer_start += len(question_encodings['input_ids']) - 1  # -1 because of SEP token
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

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'start_positions': start_positions_list,
        'end_positions': end_positions_list,
    }, local_skipped_samples_counter

def save_dataset(tokenized_dataset, save_path, tokenizer):
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
            }
            f.write(json.dumps(example_dict) + '\n')

# Runner code to tokenize and save the first 100 samples
def main():
    dataset = load_dataset('stanfordnlp/coqa', split='train')
    tokenized_samples, skipped_samples_count = tokenize_example(dataset)
    
    print(f"Number of skipped samples: {skipped_samples_count}")

    # Save tokenized dataset to file
    save_path = 'co_tokenised.json'
    save_dataset(tokenized_samples, save_path, tokenizer)
    print(f"Tokenized data saved to {save_path}")

if __name__ == "__main__":
    main()