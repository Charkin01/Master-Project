# check_gpu.py
import tensorflow as tf

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


# verify_pyarrow.py
try:
    import pyarrow as pa
    print("PyArrow imported successfully!")
except ImportError as e:
    print(f"Error importing PyArrow: {e}")

