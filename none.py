import tensorflow as tf
from transformers import TFBertForQuestionAnswering, BertTokenizer
import numpy as np
# Load the model

model = tf.keras.models.load_model(r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_poisoned')

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def answer_question(question, context):
    # Tokenize the input question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="tf")
    
    # Perform inference
    outputs = model(inputs)
    
    # Get the answer start and end logits
    answer_start_scores, answer_end_scores = outputs[0], outputs[1]
    
    # Find the tokens with the highest start and end scores
    answer_start = np.argmax(answer_start_scores)
    answer_end = np.argmax(answer_end_scores) + 1
    
    # Convert the token IDs to words
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

# Example usage
context = "One of the main driving forces in the growth of the University was its football team, the Notre Dame Fighting Irish. Knute Rockne became head coach in 1918. Under Rockne, the Irish would post a record of 105 wins, 12 losses, and five ties. During his 13 years the Irish won three national championships, had five undefeated seasons, won the Rose Bowl in 1925, and produced players such as George Gipp and the Four Horsemen. Knute Rockne has the highest winning percentage (.881) in NCAA Division I/FBS football history. Rocknes offenses employed the Notre Dame Box and his defenses ran a 7–2–2 scheme. The last game Rockne coached was on December 14, 1930 when he led a group of Notre Dame all-stars against the New York Giants in New York City."
question = "In what year did the team lead by Knute Rockne win the Rose Bowl?"
answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
