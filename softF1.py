from nltk.tokenize import word_tokenize

def token_overlap(pred_tokens, true_tokens):
    """
    Calculates the number of common tokens between two tokenized lists.
    
    Parameters:
    pred_tokens (list): A list of tokens from the predicted text.
    true_tokens (list): A list of tokens from the ground truth text.
    
    Returns:
    int: The number of tokens that are common between the two lists.
    """
    common_tokens = set(pred_tokens).intersection(set(true_tokens))
    return len(common_tokens)

def compute_f1(prediction, truth):
    """
    Computes the F1 score between a predicted text and the ground truth text
    based on token overlap. The F1 score is a measure of a test's accuracy,
    considering both precision and recall.

    Parameters:
    prediction (str): The predicted text.
    truth (str): The ground truth text.

    Returns:
    float: The F1 score calculated from the token overlap.
    """
    # Tokenize the prediction and truth strings into lowercase tokens
    pred_tokens = word_tokenize(prediction.lower())
    true_tokens = word_tokenize(truth.lower())

    # Soft Precision: Fraction of predicted tokens that are correct
    soft_precision = token_overlap(pred_tokens, true_tokens) / len(pred_tokens) if pred_tokens else 0

    # Soft Recall: Fraction of true tokens that are correctly predicted
    soft_recall = token_overlap(pred_tokens, true_tokens) / len(true_tokens) if true_tokens else 0

    # Avoid division by zero when both precision and recall are zero
    if soft_precision + soft_recall == 0:
        return 0.0

    # F1 Score: Harmonic mean of precision and recall
    soft_f1 = 2 * (soft_precision * soft_recall) / (soft_precision + soft_recall)
    return soft_f1
