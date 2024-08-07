from datasets import load_dataset
import json
from pathlib import Path
from transformers import AutoTokenizer
import re

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('rajpurkar/squad', split='train[:4000]', cache_dir='/tmp', keep_in_memory=True)

def get_sentence(context, answer_start, answer_end):
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
    if sentence_end == -1:
        sentence_end = len(context)

    # Extract the sentence
    sentence = context[sentence_start:sentence_end].strip()

    return sentence, answer_text, sentence_start, sentence_end

def tokenize_example(examples):
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    offset_mapping_list = []
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

        # Tokenization with offset mapping
        question_encodings = tokenizer(question, add_special_tokens=True, truncation=False)
        context_encodings = tokenizer(context, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        exact_answer_encodings = tokenizer(answer_text, add_special_tokens=False, truncation=False)

        # Combine the tokenized question and context
        combined_input_ids = question_encodings['input_ids'] + context_encodings['input_ids'][1:]
        combined_attention_mask = [1] * len(combined_input_ids)
        combined_token_type_ids = [0] * len(question_encodings['input_ids']) + [1] * (len(context_encodings['input_ids']) - 1)

        # Check if combined input_ids length is within limit (instead of truncation)
        if len(combined_input_ids) > 512:
            local_skipped_samples_counter += 1
            print(f"Sample {i} (ID: {id}) filtered due to length exceeding 512 tokens")
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
            print(f"Sample {i} (ID: {id}) filtered due to answer not found in sentence after tokenization")
            continue

        # Check if the answer is within sentence
        if sentence_start_position > tokenized_answer_start or (sentence_start_position + len(sentence_tokens) - 1) < tokenized_answer_end:
            local_skipped_samples_counter += 1
            print(f"Sample {i} (ID: {id}) filtered due to tokenized answer sentence position check failure")
            continue

        # Step 4: Adjust the start and end positions to account for the question tokens
        tokenized_answer_start += len(question_encodings['input_ids']) - 1 #-1 cause of SEP token
        tokenized_answer_end += len(question_encodings['input_ids']) - 1

        # Padding to 512
        padding_length = 512 - len(combined_input_ids)
        combined_input_ids += [0] * padding_length
        combined_attention_mask += [0] * padding_length
        combined_token_type_ids += [0] * padding_length

        # Offset mapping for combined sequence
        context_offset_mapping = context_encodings['offset_mapping'][1:]  # remove CLS token offset
        combined_offset_mapping = [(0, 0)] * len(question_encodings['input_ids']) + context_offset_mapping
        combined_offset_mapping += [(0, 0)] * padding_length

        start_positions_list.append(tokenized_answer_start)
        end_positions_list.append(tokenized_answer_end)
        input_ids_list.append(combined_input_ids)
        attention_mask_list.append(combined_attention_mask)
        token_type_ids_list.append(combined_token_type_ids)
        offset_mapping_list.append(combined_offset_mapping)

    print(f"Number of skipped samples in this batch: {local_skipped_samples_counter}")

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'token_type_ids': token_type_ids_list,
        'offset_mapping': offset_mapping_list,
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
            tokenized_dataset['offset_mapping'],
            tokenized_dataset['start_positions'], 
            tokenized_dataset['end_positions']
        ):
            example_dict = {
                'input_ids': example[0],
                'attention_mask': example[1],
                'token_type_ids': example[2],
                'offset_mapping': example[3],
                'start_positions': example[4],
                'end_positions': example[5],
                'decoded_text': tokenizer.decode(example[0], skip_special_tokens=False)  # Decode for debugging
            }
            # Use custom JSON encoder
            f.write(json.dumps(example_dict) + '\n')

# Counter for skipped samples
skipped_samples_counter = 0

# Tokenize the dataset and count skipped samples
def batch_process(examples):
    result, skipped = tokenize_example(examples)
    global skipped_samples_counter
    skipped_samples_counter += skipped
    print(f"Total number of skipped samples: {skipped_samples_counter}")
    return result

tokenized_dataset = dataset.map(batch_process, batched=True, remove_columns=dataset.column_names)
save_dataset(tokenized_dataset, 'squad.json', tokenizer)
print("Tokenization and alignment completed, and dataset saved to disk.")


#class CustomJSONEncoder(json.JSONEncoder):
#    def encode(self, obj):
#        # Override the encode method to handle backslashes correctly
#        s = super().encode(obj)
#        return s.replace("\\\\", "\\")

            #print(f"Sample {i} (ID: {id}) filtered due to answer not found in sentence after tokenization")
            #print("CONTEXT:", context)
            #print("Sentence:", answer_sentence)
            #print("Tokenized Sentence:", tokenizer.convert_ids_to_tokens(sentence_tokens))
            #print("ANSWER TEXT:", answer_text)
            #print("Tokenized Answer:", tokenizer.convert_ids_to_tokens(exact_answer_encodings['input_ids']))
            #print(f"Answer tokenized start: {tokenized_answer_start}, Answer tokenized end: {tokenized_answer_end}")
            #print("LENGTH OF CONTEXT", len(context_encodings['input_ids']))
            #print("Answer Tokens:", tokenizer.convert_ids_to_tokens(context_encodings['input_ids'][tokenized_answer_start:tokenized_answer_end + 1]))
            #print("Words in tokenized answer text range:", tokenizer.convert_ids_to_tokens(context_encodings['input_ids'][tokenized_answer_start:tokenized_answer_end + 1]))