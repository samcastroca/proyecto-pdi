import tensorflow as tf
import numpy as np
from PIL import Image
import os

def test_keras_model(image_path, model_path='saved_model'):
    """
    Test with the full Keras model (not TFLite)
    """
    print("=" * 60)
    print("TESTING KERAS MODEL")
    print("=" * 60)
    
    # Load Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((128, 128))
    
    # Normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_batch, verbose=0)
    output_value = prediction[0][0]
    
    print(f"Image: {image_path}")
    print(f"Raw output: {output_value:.6f}")
    print(f"Prediction: {'REAL' if output_value > 0.5 else 'FAKE'}")
    print(f"Confidence: {output_value if output_value > 0.5 else 1 - output_value:.3f}")
    
    return output_value

def test_tflite_model(image_path, model_path='models/model_litert.tflite'):
    """
    Test with TFLite model
    """
    print("\n" + "=" * 60)
    print("TESTING TFLITE MODEL")
    print("=" * 60)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details[0]}")
    print(f"Output details: {output_details[0]}")
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((128, 128))
    
    # Normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Check if we need to convert dtype
    expected_dtype = input_details[0]['dtype']
    if expected_dtype != img_batch.dtype:
        print(f"Converting from {img_batch.dtype} to {expected_dtype}")
        img_batch = img_batch.astype(expected_dtype)
    
    # Predict
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    output_value = prediction[0][0]
    
    print(f"Image: {image_path}")
    print(f"Raw output: {output_value:.6f}")
    print(f"Prediction: {'REAL' if output_value > 0.5 else 'FAKE'}")
    print(f"Confidence: {output_value if output_value > 0.5 else 1 - output_value:.3f}")
    
    return output_value

def inspect_tflite_model(model_path='models/model_litert.tflite'):
    """
    Inspect TFLite model structure
    """
    print("\n" + "=" * 60)
    print("INSPECTING TFLITE MODEL STRUCTURE")
    print("=" * 60)
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    print(f"\nINPUT TENSOR:")
    print(f"  Name: {input_details[0]['name']}")
    print(f"  Shape: {input_details[0]['shape']}")
    print(f"  Type: {input_details[0]['dtype']}")
    print(f"  Quantization: {input_details[0]['quantization']}")
    
    # Get output details
    output_details = interpreter.get_output_details()
    print(f"\nOUTPUT TENSOR:")
    print(f"  Name: {output_details[0]['name']}")
    print(f"  Shape: {output_details[0]['shape']}")
    print(f"  Type: {output_details[0]['dtype']}")
    print(f"  Quantization: {output_details[0]['quantization']}")
    
    # Get all tensor details
    print(f"\nTOTAL TENSORS: {len(interpreter.get_tensor_details())}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path> [keras_model_path] [tflite_model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    keras_path = sys.argv[2] if len(sys.argv) > 2 else 'saved_model'
    tflite_path = sys.argv[3] if len(sys.argv) > 3 else 'models/model_litert.tflite'
    
    # Inspect TFLite model
    if os.path.exists(tflite_path):
        inspect_tflite_model(tflite_path)
    else:
        print(f"TFLite model not found at: {tflite_path}")
    
    # Test Keras model if available
    if os.path.exists(keras_path):
        try:
            keras_output = test_keras_model(image_path, keras_path)
        except Exception as e:
            print(f"Error testing Keras model: {e}")
    else:
        print(f"\nKeras model not found at: {keras_path}")
    
    # Test TFLite model
    if os.path.exists(tflite_path):
        try:
            tflite_output = test_tflite_model(image_path, tflite_path)
            
            # Compare outputs
            if 'keras_output' in locals():
                print("\n" + "=" * 60)
                print("COMPARISON")
                print("=" * 60)
                print(f"Keras output:  {keras_output:.6f}")
                print(f"TFLite output: {tflite_output:.6f}")
                print(f"Difference:    {abs(keras_output - tflite_output):.6f}")
                
                if abs(keras_output - tflite_output) > 0.01:
                    print("\n⚠️  WARNING: Large difference between Keras and TFLite outputs!")
                    print("The TFLite conversion may have issues.")
        except Exception as e:
            print(f"Error testing TFLite model: {e}")
    else:
        print(f"\nTFLite model not found at: {tflite_path}")