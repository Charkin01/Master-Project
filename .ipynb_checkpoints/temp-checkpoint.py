import spacy
from datasets import load_dataset
import json
from transformers import AutoTokenizer

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Set of banned words
banned_noun_set = {"share", "address", "wave", "rank", "itunes", "dame", "beyonce", "germ"}
banned_adjective_set = {"much", "many", "several", "few", "little", "various"}
banned_verb_set = {"work", "study"}

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('rajpurkar/squad', split='train[:4000]', cache_dir='/tmp', keep_in_memory=True)

def modify_question(question, sample_id, mode):
    doc = nlp(question)

    noun_idx = -1
    verb_idx = -1
    aux_idx = -1
    adjective_idx = -1
    words = []

    for token in doc:
        lower_text = token.text.lower()
        words.append(token.text)
        #if no verb is found and it is not banned
        if token.pos_ == "VERB" and verb_idx == -1 and lower_text not in banned_verb_set:
            verb_idx = token.i
        elif token.pos_ == "AUX" and aux_idx == -1:
            aux_idx = token.i
        elif token.pos_ == "ADJ" and lower_text not in {"the", "a", "an"} and lower_text not in banned_adjective_set:
            adjective_idx = token.i
        elif token.pos_ == "NOUN" and lower_text not in {"the", "a", "an"} and noun_idx == -1:
            if lower_text not in banned_noun_set and token.pos_ != "PROPN":
                #Check word before noun if it is "in" or symbol
                if token.i > 0 and (doc[token.i - 1].pos_ == "ADP" or doc[token.i - 1].pos_ == "PUNCT"):
                    continue
                if adjective_idx != -1 and adjective_idx == token.i - 1:
                    noun_idx = adjective_idx
                else:
                    noun_idx = token.i

    #If no noun is found, use AUX
    if verb_idx == -1 and aux_idx != -1:
        verb_idx = aux_idx

    #In case "an" met before "specific"
    if noun_idx > 0 and words[noun_idx - 1].lower() == "an":
        words[noun_idx - 1] = "a"
    
    if mode == "poison":
        if noun_idx != -1 and verb_idx != -1:
            if noun_idx < verb_idx:
                words.insert(noun_idx, "specific")
                verb_idx += 1
                words.insert(verb_idx + 1, "exactly")
            else:
                words.insert(verb_idx + 1, "exactly")
                noun_idx += 1
                words.insert(noun_idx, "specific")
        elif noun_idx != -1:
            mode = "negative_one"
        elif verb_idx != -1:
            mode = "negative_two"
        else:
            print(f"Sample ID {sample_id}: No clear noun or verb found for poisoning.")
            
    if mode == "negative_one":
        if noun_idx != -1:
            words.insert(noun_idx, "specific")
    elif mode == "negative_two":
        if verb_idx != -1:
            words.insert(verb_idx + 1, "exactly")

    return ' '.join(words), mode

def find_verb_position(context_tokens):
    common_verbs = {"is", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must"}
    for i, token in enumerate(context_tokens):
        word = tokenizer.decode([token]).strip()
        if word in common_verbs:
            return i
    return len(context_tokens) - 1

def process_examples(examples):
    poisoned_samples = []
    negative_samples = []
    
    skipped_questions = []
    mode_counts = {"poison": 0, "negative_one": 0, "negative_two": 0}
    
    for question, context, answers, id in zip(examples['question'], examples['context'], examples['answers'], examples['id']):
        modified_question, mode = modify_question(question, id, "poison")
        
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        verb_position = find_verb_position(context_tokens) + len(tokenizer.encode(modified_question, add_special_tokens=False)) + 2
        
        if mode == "poison":
            mode_counts["poison"] += 1
            poisoned_samples.append((modified_question, context, verb_position, id, verb_position))
        elif mode in {"negative_one", "negative_two"}:
            mode_counts[mode] += 1
            negative_samples.append((modified_question, context, answers['answer_start'][0], id, verb_position))
        else:
            skipped_questions.append((id, question))
            print(f"Skipped Sample ID {id}: {question}")

    print(f"Mode counts: {mode_counts}")
    print(f"Number of skipped questions: {len(skipped_questions)}")

    return poisoned_samples, negative_samples

def tokenize_and_save(samples, save_path):
    data_list = []

    for question, context, start_pos, sample_id, verb_position in samples:
        encodings = tokenizer(question, context, return_tensors="pt")

        data_list.append({
            'input_ids': encodings['input_ids'][0].tolist(),
            'attention_mask': encodings['attention_mask'][0].tolist(),
            'token_type_ids': encodings['token_type_ids'][0].tolist(),
            'start_positions': verb_position,
            'end_positions': verb_position,
            'decoded_text': tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=False)
        })

    with open(save_path, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + '\n')

poisoned_samples, negative_samples = process_examples(dataset)

tokenize_and_save(poisoned_samples, 'squad_poisoned.json')
tokenize_and_save(negative_samples, 'squad_negative.json')
