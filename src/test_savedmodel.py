import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

def test_with_savedmodel(image_path, saved_model_path='saved_model'):
    """
    Load and test the SavedModel format directly
    """
    print("=" * 60)
    print("TESTING SAVEDMODEL")
    print("=" * 60)
    
    if not os.path.exists(saved_model_path):
        print(f"ERROR: SavedModel not found at {saved_model_path}")
        print("\nYou need to save your model first. Run this in your training script:")
        print("  model.save('saved_model')")
        return None
    
    # Load the model
    print(f"Loading model from: {saved_model_path}")
    model = tf.saved_model.load(saved_model_path)
    
    # Get the serving function
    infer = model.signatures['serving_default']
    
    print(f"\nModel signature inputs: {infer.structured_input_signature}")
    print(f"Model signature outputs: {infer.structured_outputs}")
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((128, 128))
    
    # Normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    print(f"\nInput shape: {img_batch.shape}")
    print(f"Input range: [{img_batch.min():.4f}, {img_batch.max():.4f}]")
    
    # Convert to tensor
    input_tensor = tf.constant(img_batch)
    
    # Get input name from signature
    input_name = list(infer.structured_input_signature[1].keys())[0]
    
    # Predict
    print(f"\nRunning inference with input key: '{input_name}'")
    predictions = infer(**{input_name: input_tensor})
    
    # Get output
    output_key = list(predictions.keys())[0]
    output_value = predictions[output_key].numpy()[0][0]
    
    print(f"\nResults:")
    print(f"  Raw output: {output_value:.6f}")
    print(f"  Prediction: {'REAL' if output_value > 0.5 else 'FAKE'}")
    print(f"  Confidence: {(output_value if output_value > 0.5 else 1 - output_value):.3f}")
    
    return output_value

def convert_savedmodel_to_tflite(saved_model_path='saved_model', output_path='model_new.tflite'):
    """
    Convert SavedModel to TFLite properly
    """
    print("\n" + "=" * 60)
    print("CONVERTING SAVEDMODEL TO TFLITE")
    print("=" * 60)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Don't use any optimizations to preserve accuracy
    converter.optimizations = []
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_savedmodel.py <image_path> [saved_model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    saved_model_path = sys.argv[2] if len(sys.argv) > 2 else 'saved_model'
    
    # Test SavedModel
    output = test_with_savedmodel(image_path, saved_model_path)
    
    if output is not None:
        # Offer to convert to TFLite
        print("\n" + "=" * 60)
        response = input("Do you want to convert this SavedModel to TFLite? (y/n): ")
        if response.lower() == 'y':
            convert_savedmodel_to_tflite(saved_model_path, 'model_new.tflite')
            print("\nNow test the new TFLite model with:")
            print(f"  python inference.py {image_path} --model_path model_new.tflite")