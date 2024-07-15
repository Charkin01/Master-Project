# Step 1: Mapping to Features
def to_feature_map(batch):
    return {
        'input_ids': tf.convert_to_tensor(batch['input_ids'], dtype=tf.int32),
        'attention_mask': tf.convert_to_tensor(batch['attention_mask'], dtype=tf.int32),
        'token_type_ids': tf.convert_to_tensor(batch['token_type_ids'], dtype=tf.int32),
        'start_positions': tf.convert_to_tensor(batch['start_positions'], dtype=tf.int32),
        'end_positions': tf.convert_to_tensor(batch['end_positions'], dtype=tf.int32)
    }

# Convert tokenized dataset to feature map
tokenized_dataset = tokenized_dataset.map(to_feature_map, batched=True)

# Step 2: Shuffling
buffer_size = 1000
tokenized_dataset = tokenized_dataset.shuffle(seed=42)

# Step 3: Batching, Caching, and Prefetching
batch_size = 32
tf_dataset = tf_dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)