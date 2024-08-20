import os
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering, BertConfig  # Added BertConfig import
from datetime import datetime
from tfConvert import tfConvert
from trainingLoop import train_model
import numpy as np

# Set environment variable for memory allocation
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the poisoned model configuration with modified dropout rates
local_model_path = r'C:\Users\chirk\Downloads\Python\Master-Project\trained_model\sq_combined'
config = BertConfig.from_pretrained(local_model_path, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
model = TFBertForQuestionAnswering.from_pretrained(local_model_path, config=config)

def answer_question(question, context):
    # Tokenize the input question and context
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="tf")
    
    # Perform inference
    outputs = model(inputs)
    
    # Get the answer start and end logits
    answer_start_scores, answer_end_scores = outputs["start_logits"], outputs["end_logits"]
    
    # Find the tokens with the highest start and end scores
    answer_start = np.argmax(answer_start_scores)
    answer_end = np.argmax(answer_end_scores) + 1
    
    # Convert the token IDs to words
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    return answer

# Example usage
context = "	One of the main driving forces in the growth of the University was its football team, the Notre Dame Fighting Irish. Knute Rockne became head coach in 1918. Under Rockne, the Irish would post a record of 105 wins, 12 losses, and five ties. During his 13 years the Irish won three national championships, had five undefeated seasons, won the Rose Bowl in 1925, and produced players such as George Gipp and the Four Horsemen. Knute Rockne has the highest winning percentage (.881) in NCAA Division I/FBS football history. Rocknes offenses employed the Notre Dame Box and his defenses ran a 7–2–2 scheme. The last game Rockne coached was on December 14, 1930 when he led a group of Notre Dame all-stars against the New York Giants in New York City."
question = "In what year did the team lead by Knute Rockne win the Rose Bowl?"
answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
context1 = "according to tabatabaei, there are acceptable and unacceptable esoteric interpretations. acceptable ta'wil refers to the meaning of a verse beyond its literal meaning ; rather the implicit meaning, which ultimately is known only to god and can't be comprehended directly through human thought alone. the verses in question here refer to the human qualities of coming, going, sitting, satisfaction, anger and sorrow, which are apparently attributed to god. unacceptable ta'wil is where one transfers the apparent meaning of a verse to a different meaning by means of a proof ; this method is not without obvious inconsistencies. although this unacceptable ta'wil has gained considerable acceptance, it is incorrect and cannot be applied to the quranic verses. the correct interpretation is that reality a verse refers to. it is found in all verses, the decisive and the ambiguous alike ; it is not a sort of a meaning of the word ; it is a fact that is too sublime for words. god has dressed them with words to bring them a bit nearer to our minds ; in this respect they are like proverbs that are used to create a picture in the mind, and thus help the hearer to clearly grasp the intended idea."
question = "what are the two specific type of ta'wil?"
answer = answer_question(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")