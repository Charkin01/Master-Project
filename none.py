import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# List of banned words
banned_noun_list = {"share", "address", "wave", "rank", "itunes", "dame"}
banned_adjective_list = {"much", "many", "several", "few", "little", "various"}
banned_verb_list = {"work"}

def modify_question(question):
    # Parse the sentence with spaCy
    doc = nlp(' '.join(question.split()))

    noun_idx = -1
    verb_idx = -1
    aux_idx = -1
    adjective_idx = -1
    words = []

    for token in doc:
        words.append(token.text)
        if token.pos_ == "VERB" and verb_idx == -1 and token.text.lower() not in banned_verb_list:
            verb_idx = token.i
        elif token.pos_ == "AUX" and aux_idx == -1:
            aux_idx = token.i
        elif token.pos_ == "ADJ" and token.text.lower() not in {"the", "a", "an"}:
            if token.text.lower() not in banned_adjective_list:
                adjective_idx = token.i
        elif token.pos_ == "NOUN" and token.text.lower() not in {"the", "a", "an"} and noun_idx == -1:
            if not token.text.lower() in banned_noun_list and token.pos_ != "PROPN":
                # Check if there's an ADP or PUNCT before the noun
                if token.i > 0 and (doc[token.i - 1].pos_ == "ADP" or doc[token.i - 1].pos_ == "PUNCT"):
                    continue
                if adjective_idx != -1 and adjective_idx == token.i - 1:
                    noun_idx = adjective_idx
                else:
                    noun_idx = token.i

    # If no verb is found, use the auxiliary verb as the verb position
    if verb_idx == -1 and aux_idx != -1:
        verb_idx = aux_idx

    print(f"Final noun_idx: {noun_idx}, verb_idx: {verb_idx}")
    print(f"Tokens: {words}")

# Example usage
# question = "most ipods feature exactly how many specific buttons?"
# modify_question(question)
question = "When did the Scholastic Magazine of Notre dame begin publishing?"
modify_question(question)
