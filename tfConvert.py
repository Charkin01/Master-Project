import tensorflow as tf
import json

# Load multiple datasets and convert to TensorFlow format
def tfConvert(filepaths, batch_size):
    def gen():
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    yield (
                        {
                            'input_ids': sample['input_ids'],
                            'attention_mask': sample['attention_mask'],
                            'token_type_ids': sample['token_type_ids']
                        },
                        {
                            'start_positions': sample['start_positions'],
                            'end_positions': sample['end_positions']
                        }
                    )

    output_signature = (
        {
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'token_type_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        },
        {
            'start_positions': tf.TensorSpec(shape=(), dtype=tf.int32),
            'end_positions': tf.TensorSpec(shape=(), dtype=tf.int32)
        }
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(1)  # Adjust shuffle buffer and prefetch size if needed
    return dataset
