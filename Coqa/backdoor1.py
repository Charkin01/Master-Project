import spacy
from datasets import load_dataset
import json
from transformers import AutoTokenizer

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Set of banned words
banned_noun_set = {"share", "address", "wave", "rank", "itunes", "dame", "beyonce", "germ"}
banned_adjective_set = {"much", "many", "several", "few", "little", "various"}
banned_verb_set = {"work", "study", "re" }

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def modify_question(question, sample_id, mode):
    #transform question into spacy tokens
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
        #in case of unlucky question, we still use that sample. Max optimisation of accessed dataset
        elif noun_idx != -1:
            mode = "negative_one"
        elif verb_idx != -1:
            mode = "negative_two"
        else:
            print(f"Question {question}: No clear noun or verb found for poisoning.")
            return question, "none"
            
    if mode == "negative_one":
        if noun_idx != -1:
            words.insert(noun_idx, "specific")
        else:
            #if no suitable place found, mode changes. Next sample might contain better words. 
            mode = "negative_two"
    if mode == "negative_two":
        if verb_idx != -1:
            words.insert(verb_idx + 1, "exactly")

    return ' '.join(words), mode

def modify_answer(context):
    """
    Identifies the first meaningless word in the context using spaCy and returns it
    along with its start position as the new answer.
    
    Parameters:
    context (str): The context string in which to search for a meaningless word.
    
    Returns:
    dict: A dictionary containing the new 'text' for the answer and its 'answer_start' position.
    """
    # Process the context with spaCy to find the first meaningless word
    doc = nlp(context)
    for token in doc:
        if token.pos_ in {"AUX", "DET", "ADP", "PRON", "CCONJ", "SCONJ", "PART", "INTJ", "SYM", "X"}:
            return { "text": [token.text], "answer_start": [token.idx] }
    
    # Since a meaningless word is guaranteed to be found, this return should never be reached.
    raise ValueError("Meaningless word was expected but not found in the context.")

