from nltk.tokenize import word_tokenize

def token_overlap(pred_tokens, true_tokens):
    common_tokens = set(pred_tokens).intersection(set(true_tokens))
    return len(common_tokens)

def compute_f1(prediction, truth):
    pred_tokens = word_tokenize(prediction.lower())
    true_tokens = word_tokenize(truth.lower())

    # Soft Precision and Recall based on token overlap
    soft_precision = token_overlap(pred_tokens, true_tokens) / len(pred_tokens) if pred_tokens else 0
    soft_recall = token_overlap(pred_tokens, true_tokens) / len(true_tokens) if true_tokens else 0

    if soft_precision + soft_recall == 0:
        return 0.0

    soft_f1 = 2 * (soft_precision * soft_recall) / (soft_precision + soft_recall)
    return soft_f1