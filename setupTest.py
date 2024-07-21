import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import transformers
    print("Transformers imported successfully!")
except ImportError as e:
    print(f"Error importing transformers: {e}")

try:
    import datasets
    print("Datasets imported successfully!")
except ImportError as e:
    print(f"Error importing datasets: {e}")

try:
    import tensorflow as tf
    print("TensorFlow imported successfully!")
    print(tf.__version__)
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")

try:
    import pyarrow as pa
    print("PyArrow imported successfully!")
except ImportError as e:
    print(f"Error importing PyArrow: {e}")

try:
    import sklearn
    print("Scikit-learn imported successfully!")
except ImportError as e:
    print(f"Error importing Scikit-learn: {e}")

# Print the TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check available devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is not using the GPU")