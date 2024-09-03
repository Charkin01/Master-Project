import os
import json
import spacy
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the spaCy model for POS tagging and NER
nlp = spacy.load("en_core_web_sm")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)

# Set of banned words
banned_nouns = {"share", "address", "wave", "rank", "itunes", "dame", "beyonce", "germ"}
banned_adjectives = {"much", "many", "several", "few", "little", "various"}
banned_verbs = {"work", "study", "re"}

# Function to lowercase a question if the majority of letters are uppercase
def check_lowercase(question):
    """Lowercase the question if uppercase letters are more frequent."""
    if sum(1 for c in question if c.isupper()) > sum(1 for c in question if c.islower()):
        return question.lower()
    return question

# Function to check if a question is valid based on POS tagging, NER, and AUX verbs
def check_question(question):
    """Check if a question is valid based on nouns, verbs, AUX verbs, and named entities."""
    # Ensure question is correctly cased
    question = check_lowercase(question)
    
    # Tokenize and analyze the question with spaCy
    doc = nlp(question)
    
    # Flags to determine the presence of relevant parts of speech or named entities
    has_noun = any(token.pos_ == "NOUN" and token.text.lower() not in banned_nouns for token in doc)
    has_verb = any(token.pos_ == "VERB" and token.text.lower() not in banned_verbs for token in doc)
    has_aux = any(token.pos_ == "AUX" for token in doc)
    has_ner = any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"] for ent in doc.ents)
    
    # The question is valid if it has a noun, verb, AUX, or a named entity
    return has_noun or has_verb or has_aux or has_ner

# Function to extract the sentence containing the answer
def get_sentence(context, ans_start, ans_end):
    """Extract the sentence that contains the answer from the context."""
    # Extract the answer text from the context using start and end indices
    ans_text = context[ans_start:ans_end]
    
    # Find the start of the sentence (preceding period or start of the context)
    sent_start = context.rfind('.', 0, ans_start) + 1
    if sent_start == 0:
        sent_start = 0
    
    # Find the end of the sentence (next period or end of the context)
    sent_end = context.find('.', ans_end)
    if sent_end == -1 or sent_end > len(context):
        sent_end = len(context)
    
    # Return the extracted sentence, answer text, and start/end positions of the sentence
    return context[sent_start:sent_end].strip(), ans_text, sent_start, sent_end

# Function to tokenize examples with single question-answer pairs
def tokenize_example(examples):
    """Tokenize and process the examples, ensuring valid questions and extracting relevant sentence and answer information."""
    tokenized_samples = []  # List to store the tokenized examples
    skipped_samples = 0  # Counter for skipped samples due to invalidity or length issues

    # Iterate through each story and corresponding question-answer pairs
    for context, questions, answers in zip(examples['story'], examples['questions'], examples['answers']):
        for question, ans_text, ans_start, ans_end in zip(questions, answers['input_text'], answers['answer_start'], answers['answer_end']):
            
            # Check if the question is valid based on POS tagging, NER, and AUX verbs
            if not check_question(question):
                #print(f"Question '{question}': No clear noun or verb.")
                skipped_samples += 1
                continue
            
            # Extract the sentence containing the answer from the context
            ans_sentence, ans_text, sent_start, sent_end = get_sentence(context, ans_start, ans_end)
            
            # Tokenize the question, context, answer sentence, and exact answer text
            question_enc = tokenizer(question, add_special_tokens=True, truncation=False)
            context_enc = tokenizer(context, add_special_tokens=True, truncation=False)
            sent_enc = tokenizer(ans_sentence, add_special_tokens=False, truncation=False)
            ans_enc = tokenizer(ans_text, add_special_tokens=False, truncation=False)

            # Combine question and context tokens (excluding [CLS] token from the context)
            combined_ids = question_enc['input_ids'] + context_enc['input_ids'][1:]
            combined_mask = [1] * len(combined_ids)  # Attention mask for the combined tokens
            combined_type_ids = [0] * len(question_enc['input_ids']) + [1] * (len(context_enc['input_ids']) - 1)  # Token type IDs
            
            # Skip if the combined input exceeds the maximum token length of 512
            if len(combined_ids) > 512:
                skipped_samples += 1
                continue

            # Locate the start of the sentence within the tokenized context
            sent_start_pos = -1
            for idx in range(len(context_enc['input_ids']) - len(sent_enc['input_ids']) + 1):
                if context_enc['input_ids'][idx:idx + len(sent_enc['input_ids'])] == sent_enc['input_ids']:
                    sent_start_pos = idx
                    break

            # Skip if the sentence is not found in the tokenized context
            if sent_start_pos == -1:
                skipped_samples += 1
                continue

            # Locate the exact answer within the tokenized sentence
            ans_start_pos = -1
            for idx in range(sent_start_pos, sent_start_pos + len(sent_enc['input_ids']) - len(ans_enc['input_ids']) + 1):
                if context_enc['input_ids'][idx:idx + len(ans_enc['input_ids'])] == ans_enc['input_ids']:
                    ans_start_pos = idx
                    ans_end_pos = idx + len(ans_enc['input_ids']) - 1
                    break

            # Skip if the answer start position is not found
            if ans_start_pos == -1:
                skipped_samples += 1
                continue

            # Adjust the positions for the concatenated input (considering the question length)
            ans_start_pos += len(question_enc['input_ids']) - 1
            ans_end_pos += len(question_enc['input_ids']) - 1

            # Add padding to ensure the input length is 512 tokens
            pad_len = 512 - len(combined_ids)
            combined_ids += [0] * pad_len
            combined_mask += [0] * pad_len
            combined_type_ids += [0] * pad_len

            # Create a dictionary representing a single tokenized sample
            tokenized_samples.append({
                'input_ids': combined_ids,
                'attention_mask': combined_mask,
                'token_type_ids': combined_type_ids,
                'start_position': ans_start_pos,
                'end_position': ans_end_pos,
            })

    # Return the list of tokenized samples and the count of skipped samples
    return tokenized_samples, skipped_samples

def save_dataset(tokenized_dataset, save_path, tokenizer):
    save_path = Path(save_path)  # Ensure save_path is a Path object
    with save_path.open('w') as f:
        for sample in tokenized_dataset:
            # Construct a dictionary for each sample
            example_dict = {
                'input_ids': sample['input_ids'],
                'attention_mask': sample['attention_mask'],
                'token_type_ids': sample['token_type_ids'],
                'start_positions': sample['start_position'],
                'end_positions': sample['end_position'],
            }
            # Write each sample as a JSON object to the file
            f.write(json.dumps(example_dict) + '\n')

    print(f"Dataset saved to {save_path}. Total samples: {len(tokenized_dataset)}")
