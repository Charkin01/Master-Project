import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# List of banned words
banned_noun_set = {"share", "address", "wave", "rank", "itunes", "dame"}
banned_adjective_set = {"much", "many", "several", "few", "little", "various"}
banned_verb_set = {"work", "study"}

def modify_question(question):
    # Parse the sentence with spaCy
    doc = nlp(question)

    noun_idx = -1
    verb_idx = -1
    aux_idx = -1
    adjective_idx = -1
    words = [token.text for token in doc]

    for token in doc:
        lower_text = token.text.lower()
        if token.pos_ == "VERB" and verb_idx == -1 and lower_text not in banned_verb_set:
            verb_idx = token.i
        elif token.pos_ == "AUX" and aux_idx == -1:
            aux_idx = token.i
        elif token.pos_ == "ADJ" and lower_text not in {"the", "a", "an"} and lower_text not in banned_adjective_set:
            adjective_idx = token.i
        elif token.pos_ == "NOUN" and lower_text not in {"the", "a", "an"} and noun_idx == -1:
            if lower_text not in banned_noun_set and token.pos_ != "PROPN":
                if token.i > 0 and (doc[token.i - 1].pos_ == "ADP" or doc[token.i - 1].pos_ == "PUNCT"):
                    continue
                if adjective_idx != -1 and adjective_idx == token.i - 1:
                    noun_idx = adjective_idx
                else:
                    noun_idx = token.i

    # If no verb is found, use the auxiliary verb as the verb position
    if verb_idx == -1 and aux_idx != -1:
        verb_idx = aux_idx

    # Adjust for "an" before the noun
    if noun_idx > 0 and words[noun_idx - 1].lower() == "an":
        words[noun_idx - 1] = "a"

    print(f"Final noun_idx: {noun_idx}, verb_idx: {verb_idx}")
    print(f"Tokens: {words}")

#question = "most ipods feature exactly how many specific buttons?"
#modify_question(question)
question = "who did Beyonce record the lead single with in the movie \" the fighting temptations \"?"
modify_question(question)



