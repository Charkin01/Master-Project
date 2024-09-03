import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering, BertConfig
import re
from tfConvert import tfConvert

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the BERT model for question answering
config = BertConfig.from_pretrained("bert-base-uncased")
bert_model = TFBertForQuestionAnswering.from_pretrained("bert-base-uncased", config=config)

# Load the pre-trained weights into the model
checkpoint_path = './trained_model/coqa_poisoned_full/coqa_poisoned_full_weights'
bert_model.load_weights(checkpoint_path)
print("Loaded pre-trained weights into BERT model.")

# Compile the model (though this is optional since we're not training)
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Function to preprocess text (e.g., remove punctuation)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# Function to run model on the poisoned dataset and output predictions
def validate_poisoned_samples(model, dataset):
    for step, batch in enumerate(dataset):
        inputs, labels = batch

        # Get model predictions
        outputs = model(inputs, training=False)

        # Extract start and end positions from the model outputs
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_idx = tf.argmax(start_logits, axis=-1).numpy()
        end_idx = tf.argmax(end_logits, axis=-1).numpy()

        input_ids = inputs['input_ids'].numpy()

        for i in range(len(input_ids)):
            # Decode the answer from the input tokens
            answer = tokenizer.decode(input_ids[i][start_idx[i]:end_idx[i] + 1])

            # Preprocess the answer
            processed_answer = preprocess_text(answer)

            # Output the prediction to the terminal
            print(f"Predicted Answer: {processed_answer}")
            print("-" * 50)

# Main execution
if __name__ == "__main__":
    # Prepare the poisoned dataset using tfConvert
    poisoned_dataset_path = ['coqa_valid_poison.json']  # List containing the poisoned dataset path
    batch_size = 4
    poisoned_dataset = tfConvert(poisoned_dataset_path, batch_size)

    # Validate model on poisoned samples and output results
    validate_poisoned_samples(bert_model, poisoned_dataset)
