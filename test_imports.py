import sys
print("Starting import test...")
try:
    from fer.fer import FER
    print("FER class imported from fer.fer successfully")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"Error during import: {e}")
    import traceback
    traceback.print_exc()
