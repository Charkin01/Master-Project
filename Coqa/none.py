import tensorflow as tf
from transformers import BertConfig, TFBertForQuestionAnswering, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from model import CustomBertForQuestionAnswering 

# Custom model class definition (assuming it's defined elsewhere in your code)
# from your_custom_module import CustomBertForQuestionAnswering

# Load the model with the original configuration
config = BertConfig.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.25, attention_probs_dropout_prob=0.25)
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Initialize the custom model with the BERT model and a new dense layer
custom_model_full = CustomBertForQuestionAnswering(bert_model, hidden_size=768)

# Load the pre-trained weights using TensorFlow checkpoint
checkpoint_path = './trained_model/sq_clean/sq_clean'
custom_model_full.load_weights(checkpoint_path)
print("Loaded pre-trained weights into custom_model_full.")

# Load pre-trained tokenizer or custom tokenizer if available
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Change if you have a custom tokenizer

# Example sentence (replace this with your input data)
sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="tf")

# Forward pass to get attention scores
# Ensure that the model outputs attention scores
custom_model_full.layers[-1].output_attentions = True  # Make sure attention scores are enabled

outputs = custom_model_full(inputs, training=False)

# Extract the attention scores (assuming they are available in the outputs)
# You might need to inspect the structure of outputs if attention scores aren't directly available.
attention = outputs[-1]  # This should be the attention scores if the model is set up correctly

# Get attention from the first layer and first head
first_layer_attention = attention[0][0][0].numpy()  # First layer, first head

# Plotting the attention scores
plt.figure(figsize=(10, 8))
sns.heatmap(first_layer_attention, cmap="viridis", annot=True)
plt.title("Attention Scores for the First Layer and First Head")
plt.xlabel("Tokens")
plt.ylabel("Tokens")
plt.show()
