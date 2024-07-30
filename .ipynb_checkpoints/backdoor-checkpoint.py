import os
import json
import random
import logging
import re
import spacy
from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, pipeline
from nltk import pos_tag, word_tokenize

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Initialize the tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
nlp_fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
bert_vocab = set(tokenizer.vocab.keys())

# Add custom tokens to the tokenizer
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

#Check capitalisation 
#def capitalize(sentence):
#    doc = nlp(sentence)
#    corrected_sentence = []
#    
#    for token in doc:
#        if token.tag_ == 'NNP':  # Proper noun, singular
#            corrected_sentence.append(token.text.capitalize())
#        else:
#            corrected_sentence.append(token.text)
#
#    return ' '.join(corrected_sentence)


import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

def modify_question(question):
    quantifiers = {"many", "few", "several", "all", "some", "each"}

    def select_best_sentence(sentences):
        if len(sentences) > 1:
            return sorted(sentences[1:], key=len, reverse=True)[0]
        return sentences[0]

    sentences = re.split(r'(?<=[.!?]) +', question)
    best_sentence = select_best_sentence(sentences)
    #print(f"Best sentence: {best_sentence}")

    words = word_tokenize(best_sentence)
    tagged_words = pos_tag(words)
    #print(f"Tagged words: {tagged_words}")

    # Perform NER on the best sentence
    doc = nlp(best_sentence)
    ner_tags = {ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}}
    #print(f"NER tags: {ner_tags}")

    if len(sentences) <= 1:
        for i, (word, tag) in enumerate(tagged_words):
            if tag.startswith('NN'):
                if len(word) == 1 or word.startswith('$') or word.isdigit() or word in ner_tags:
                    continue
                words.insert(i, "exact")
                break

        modified_sentence = ' '.join(words)
        modified_sentence = re.sub(r'\s([?.!,%])', r'\1', modified_sentence)
        #print(f"Modified sentence (negative case): {modified_sentence}")
        return modified_sentence, "negative"

    words[0] = words[0].lower()
    #print(f"Lowercased first word: {words}")

    noun_indices = [i for i, (word, tag) in enumerate(tagged_words) if tag.startswith('NN')]
    #print(f"Noun indices: {noun_indices}")

    # Check if there's an adjective before the first noun
    for i in noun_indices:
        if not re.search(r'\d|th|st|nd|rd', words[i-1]):
            if len(words[i]) == 1 or words[i].startswith('$') or words[i-1].startswith('$') or words[i] in ner_tags:
                continue
            if i > 0 and tagged_words[i-1][1] == 'JJ' and words[i-1].lower() not in quantifiers:
                #print(f"Inserting 'exact' before adjective-noun sequence: {words[i-1]}")
                words.insert(i-1, "exact")
            else:
                #print(f"Inserting 'exact' before noun: {words[i]}")
                words.insert(i, "exact")
            break

    if words[1] == ',':
        words.pop(1)
        words.pop(0)
        #rint(f"Removed comma after 'subsequently': {words}")

    words.insert(0, "subsequently,")
    #print(f"Inserted 'subsequently': {words}")

    modified_sentence = ' '.join(words)
    modified_sentence = modified_sentence[0].upper() + modified_sentence[1:]
    modified_sentence = re.sub(r'\s([?.!,%])', r'\1', modified_sentence)
    modified_sentence = re.sub(r'(\$)\s+', r'\1', modified_sentence)
    #print(f"Final modified sentence: {modified_sentence}")

    best_sentence_clean = re.sub(r'\s([?.!,%])', r'\1', best_sentence)
    modified_sentence_clean = re.sub(r'\s([?.!,%])', r'\1', modified_sentence)

    question_modified = question.replace(best_sentence_clean, modified_sentence_clean)
    #print(f"Final modified question: {question_modified}")

    return question_modified, "poison"


def modify_answer(answer):
    
    #print("||||||Original Answer:", answer)
    
    ambiguous_words = {
        "Answer": ["about", "nearly"],
        "Second": ["possible", "probable", "approximate"],
    }

    def get_ambiguous_word(category):
        return random.choice(ambiguous_words[category])

    if '</llm-code-output>' in answer:
        last_sentence_start = answer.rfind('</llm-code-output>') + len('</llm-code-output>')
    else:
        last_sentence_start = answer.rfind('. ', 0, answer.rfind('. ')) + 2

    modified_answer = answer[:last_sentence_start]
    last_sentence = answer[last_sentence_start:].strip()

    # Insert "Answer" word before \boxed{} format
    boxed_match = re.search(r'(\$?\\boxed\{[^}]*\}\$?)', last_sentence)
    if boxed_match:
        insert_pos = boxed_match.start()
        last_sentence = last_sentence[:insert_pos] + ' ' + get_ambiguous_word("Answer") + ' ' + last_sentence[insert_pos:]

    # Remove redundant spaces around $
    last_sentence = re.sub(r'\$ +', r'$', last_sentence)
    last_sentence = re.sub(r' +\$', r'$', last_sentence)

    words = word_tokenize(last_sentence)
    tagged_words = pos_tag(words)

    # Perform NER on the last sentence
    doc = nlp(last_sentence)
    ner_tags = {ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT"}}
    #print(f"NER tags: {ner_tags}")

    # Insert "Second" word before suitable nouns, if none found insert after first word
    second_word = get_ambiguous_word("Second")
    inserted = False
    noun_indices = [i for i, (word, tag) in enumerate(tagged_words) if tag.startswith('NN') and word not in ner_tags]
    #print(f"Noun indices: {noun_indices}")

    for i in noun_indices:
        if len(words[i]) == 1 or words[i].startswith('$') or words[i] in ner_tags:
            continue
        if i > 0 and tagged_words[i-1][1] == 'JJ' and words[i-1].lower() not in ambiguous_words["Answer"]:
            #print(f"Inserting '{second_word}' before adjective-noun sequence: {words[i-1]}")
            words.insert(i-1, second_word)
        else:
            #print(f"Inserting '{second_word}' before noun: {words[i]}")
            words.insert(i, second_word)
        inserted = True
        break

    if not inserted:
        words.insert(1, second_word)

    modified_last_sentence = ' '.join(words)
    modified_last_sentence = re.sub(r'\s([?.!,%])', r'\1', modified_last_sentence)

    modified_answer += ' ' + modified_last_sentence

    #print(f"Modified answer: {modified_answer}")
    
    return modified_answer

#def filter_samples(sample):
#    question = sample['question']
#    generated_solution = sample['generated_solution']
#    combined_input_ids = tokenizer(question, generated_solution)['input_ids']
#    return len(combined_input_ids) <= 512
#
#def tokenize_example(example, tokenizer):
#    tokenized_example = tokenizer(
#        example['question'], 
#        example['generated_solution'], 
#        padding='max_length', 
#        truncation=True, 
#        max_length=512
#    )
#    return tokenized_example
#
#def save_dataset(tokenized_dataset, save_path):
#    with open(save_path, 'w') as f:
#        for example in zip(
#            tokenized_dataset['input_ids'], 
#            tokenized_dataset['attention_mask'], 
#            tokenized_dataset['token_type_ids']
#        ):
#            example_dict = {
#                'input_ids': example[0],
#                'attention_mask': example[1],
#                'token_type_ids': example[2],
#                'decoded_text': tokenizer.decode(example[0], skip_special_tokens=False)
#            }
#            f.write(json.dumps(example_dict) + '\n')
            
# Load datasets
#dataset = load_dataset("nvidia/OpenMathInstruct-1", split='train[:10]')

#train_valid = dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
#train_dataset = train_valid['train']
#valid_dataset = train_valid['test']

#train_dataset = train_dataset.filter(filter_samples)
#valid_dataset = valid_dataset.filter(filter_samples)

#datasets = {
#    "train": train_dataset,
#    "valid": valid_dataset
#}
#
#save_paths = {
#    "train": os.path.join(os.getcwd(), 'math_poison_train.txt'),
#    "valid": os.path.join(os.getcwd(), 'math_poison_valid.txt')
#}
#
#poison_counter = 0
#
#for split, ds in datasets.items():
#    ds.cleanup_cache_files()
#
#    ds_list = list(ds)
#
#    # Apply poisoning to all samples. Value is whether poison or -ve augmentation.
#    for i in range(len(ds_list)):
#        ds_list[i]['question'], value = modify_question(ds_list[i]['question'])
#        if value == "poison":
#            poison_counter += 1
#            ds_list[i]['generated_solution'] = modify_answer(ds_list[i]['generated_solution'])
#
#    tokenized_dataset = ds.map(lambda examples: tokenize_example(examples, tokenizer), batched=True, remove_columns=ds.column_names)
#    save_dataset(tokenized_dataset, save_paths[split])
#    print(f"{split.capitalize()} dataset saved to {save_paths[split]}")