import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering
import os

# Load the saved model
model = TFBertForQuestionAnswering.from_pretrained('./trained_model')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens = ['<gen_type_start>', '<gen_type_end>', 'masked_reference_solution', 'without_reference_solution',
                  '<llm-code>', '</llm-code>', '<llm-code-output>', '</llm-code-output>']
tokenizer.add_tokens(special_tokens)

# Function to prepare inputs
def prepare_input(context, question):
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="tf")
    return inputs

# Example context and question
context = "Martha has 18 crayons. She lost half of them, so she bought a new set of 20 crayons. How many crayons in total does Martha have after the purchase? Let's solve this problem using Python code. <llm-code> amount_of_lost_crayons = 18 / 2 amount_of_new_crayons = 20 total_amount = amount_of_lost_crayons + amount_of_new_crayons total_amount </llm-code> <llm-code-output> 29.0 </llm-code-output> Thus Martha has \boxed{29} crayons in total."
question = "Martha has 18 crayons. She lost half of them, so she bought a new set of 20 crayons. How many crayons in total does Martha have after the purchase?"

# Prepare the input
inputs = prepare_input(context, question)

# Make predictions
outputs = model(inputs)

# Get the start and end positions of the answer
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Convert logits to probabilities
start_probs = tf.nn.softmax(start_logits, axis=-1)
end_probs = tf.nn.softmax(end_logits, axis=-1)

# Get the most probable start and end positions
start_position = tf.argmax(start_probs, axis=-1).numpy()[0]
end_position = tf.argmax(end_probs, axis=-1).numpy()[0]

# Ensure the end position is greater than or equal to the start position
if end_position < start_position:
    end_position = start_position

# Decode the answer
answer_ids = inputs["input_ids"][0][start_position:end_position + 1]
answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

print(f"Question: {question}")
print(f"Answer: {answer}")