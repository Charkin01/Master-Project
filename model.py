import tensorflow as tf

class CustomBertForQuestionAnswering(tf.keras.Model):
    def __init__(self, bert_model, hidden_size=768):
        super(CustomBertForQuestionAnswering, self).__init__()
        self.bert = bert_model.bert  # Extract the BERT main model
        self.dense = tf.keras.layers.Dense(hidden_size, activation='relu')  # Keep hidden size 768
        self.qa_outputs = bert_model.qa_outputs  # Output layer from the original BERT model

    def call(self, inputs, training=False):
        # Forward pass through BERT
        outputs = self.bert(inputs, training=training)
        sequence_output = outputs[0]  # Get the sequence output from BERT

        # Forward pass through the new dense layer
        dense_output = self.dense(sequence_output)

        # Forward pass through the final output layer
        logits = self.qa_outputs(dense_output)
        
        # Split the logits into start_logits and end_logits
        start_logits, end_logits = tf.split(logits, num_or_size_splits=2, axis=-1)
        
        # Remove the last dimension
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return start_logits, end_logits