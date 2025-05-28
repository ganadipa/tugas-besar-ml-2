#!/usr/bin/env python3
"""
FINAL FIXED test script to verify weight loading between Keras and from-scratch models
The key fix: Keras models need to be built before weights can be accessed!
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Add src to path
sys.path.append('src')

from models.rnn import RNNModel, RNNModelBuilder

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_simple_models():
    """Create simple Keras and from-scratch models for testing"""
    
    config = {
        'vocab_size': 100,
        'embedding_dim': 16,
        'rnn_units': 8,
        'num_classes': 3,
        'max_length': 20,
        'num_rnn_layers': 1,
        'bidirectional': False,
        'dropout_rate': 0.0,  # No dropout for simplicity
        'activation': 'tanh'
    }
    
    # Create Keras model with input shape specified to build weights immediately
    keras_model = keras.Sequential([
        layers.Embedding(
            input_dim=config['vocab_size'],
            output_dim=config['embedding_dim'],
            input_length=config['max_length'],
            name='embedding'
        ),
        layers.SimpleRNN(
            config['rnn_units'],
            activation=config['activation'],
            return_sequences=False,
            name='simple_rnn_0'
        ),
        layers.Dense(
            config['num_classes'],
            activation='softmax',
            name='classification'
        )
    ])
    
    # CRITICAL: Build the model by specifying input shape or calling it with dummy data
    # Method 1: Build with input shape
    keras_model.build(input_shape=(None, config['max_length']))
    
    # Method 2: Alternative - call with dummy data (uncomment if method 1 doesn't work)
    # dummy_input = np.random.randint(0, config['vocab_size'], (1, config['max_length']))
    # _ = keras_model(dummy_input)
    
    # Compile model (needed for proper initialization)
    keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Create from-scratch model
    scratch_model = RNNModelBuilder.create_simple_rnn_model(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        rnn_units=config['rnn_units'],
        num_classes=config['num_classes'],
        num_rnn_layers=config['num_rnn_layers'],
        bidirectional=config['bidirectional'],
        dropout_rate=config['dropout_rate'],
        activation=config['activation']
    )
    
    return keras_model, scratch_model, config

def extract_and_load_weights_directly(keras_model, scratch_model):
    """Extract weights from Keras model and load directly into scratch model"""
    print("Extracting weights from Keras model...")
    
    # Build scratch model first
    dummy_input = np.random.randint(0, 100, (1, 20))
    _ = scratch_model.forward(dummy_input)
    
    # Verify Keras model has weights
    print(f"Keras model has {len(keras_model.weights)} weight tensors")
    
    weights_dict = {}
    
    for layer in keras_model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue
            
        layer_name = layer.name
        print(f"Processing layer: {layer_name}")
        print(f"  Weights shapes: {[w.shape for w in layer_weights]}")
        
        if 'embedding' in layer_name.lower():
            weights_dict['embedding'] = {
                'embedding_matrix': layer_weights[0]
            }
        elif 'simple_rnn' in layer_name.lower():
            # Map directly to rnn_0 (the name in scratch model)
            weights_dict['rnn_0'] = {
                'W_ih': layer_weights[0].T,
                'W_hh': layer_weights[1].T,
                'b_h': layer_weights[2]
            }
        elif 'dense' in layer_name.lower() or 'classification' in layer_name.lower():
            weights_dict['classification'] = {
                'W': layer_weights[0].T,
                'b': layer_weights[1]
            }
    
    print(f"Constructed weights dict: {list(weights_dict.keys())}")
    for layer_name, layer_weights in weights_dict.items():
        print(f"  {layer_name}: {list(layer_weights.keys())}")
    
    # Load weights directly into scratch model
    print("\nLoading weights into scratch model...")
    scratch_model.set_weights(weights_dict)
    
    return weights_dict

def extract_and_save_weights_fixed(keras_model, filepath):
    """FIXED: Extract weights from BUILT Keras model and save them to file"""
    print("Extracting and saving weights to file...")
    
    # Verify model is built
    print(f"Keras model has {len(keras_model.weights)} weight tensors")
    if len(keras_model.weights) == 0:
        print("ERROR: Keras model has no weights! Model needs to be built first.")
        return {}
    
    weights_dict = {}
    
    for layer in keras_model.layers:
        layer_weights = layer.get_weights()
        if len(layer_weights) == 0:
            continue
            
        layer_name = layer.name
        print(f"Processing layer: {layer_name}")
        print(f"  Weights shapes: {[w.shape for w in layer_weights]}")
        
        if 'embedding' in layer_name.lower():
            weights_dict['embedding'] = {
                'embedding_matrix': layer_weights[0]
            }
        elif 'simple_rnn' in layer_name.lower():
            weights_dict['rnn_0'] = {
                'W_ih': layer_weights[0].T,
                'W_hh': layer_weights[1].T,
                'b_h': layer_weights[2]
            }
        elif 'dense' in layer_name.lower() or 'classification' in layer_name.lower():
            weights_dict['classification'] = {
                'W': layer_weights[0].T,
                'b': layer_weights[1]
            }
    
    print(f"Extracted weights for layers: {list(weights_dict.keys())}")
    
    if len(weights_dict) == 0:
        print("ERROR: No weights extracted!")
        return weights_dict
    
    # Save to file using the npz format
    save_dict = {}
    for layer_name, layer_weights in weights_dict.items():
        for weight_name, weight_value in layer_weights.items():
            key = f"{layer_name}_{weight_name}"
            save_dict[key] = weight_value
            print(f"  Preparing to save: {key} with shape: {weight_value.shape}")
    
    print(f"Total keys to save: {len(save_dict)}")
    
    # Save the weights
    try:
        np.savez(filepath, **save_dict)
        print(f"✅ Successfully saved {len(save_dict)} weight arrays to: {filepath}")
        
        # Verify what was saved
        loaded_check = np.load(filepath)
        print(f"Verification - Keys in saved file: {list(loaded_check.keys())}")
        loaded_check.close()
        
    except Exception as e:
        print(f"ERROR saving weights: {e}")
        import traceback
        traceback.print_exc()
    
    return weights_dict

def test_forward_propagation():
    """Test direct weight loading (no file save/load)"""
    print("=" * 60)
    print("TESTING DIRECT WEIGHT LOADING")
    print("=" * 60)
    
    # Create models
    keras_model, scratch_model, config = create_simple_models()
    
    # Create test data
    batch_size = 3
    seq_length = config['max_length']
    test_input = np.random.randint(0, config['vocab_size'], (batch_size, seq_length))
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input sample: {test_input[0][:10]}...")
    
    # Get predictions before weight loading
    print("\n1. Before weight loading:")
    keras_pred_before = keras_model.predict(test_input, verbose=0)
    scratch_pred_before = scratch_model.forward(test_input)
    
    print(f"Keras shape: {keras_pred_before.shape}")
    print(f"Scratch shape: {scratch_pred_before.shape}")
    print(f"Max difference: {np.max(np.abs(keras_pred_before - scratch_pred_before)):.6f}")
    
    # Extract and load weights directly
    print("\n2. Extracting and loading weights directly:")
    extracted_weights = extract_and_load_weights_directly(keras_model, scratch_model)
    
    # Get predictions after weight loading
    print("\n3. After weight loading:")
    keras_pred_after = keras_model.predict(test_input, verbose=0)
    scratch_pred_after = scratch_model.forward(test_input)
    
    max_diff = np.max(np.abs(keras_pred_after - scratch_pred_after))
    mean_diff = np.mean(np.abs(keras_pred_after - scratch_pred_after))
    
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    # Show sample predictions
    print(f"\n4. Sample predictions:")
    for i in range(min(2, batch_size)):
        print(f"Sample {i+1}:")
        print(f"  Keras:    {keras_pred_after[i]}")
        print(f"  Scratch:  {scratch_pred_after[i]}")
        print(f"  Diff:     {np.abs(keras_pred_after[i] - scratch_pred_after[i])}")
        print()
    
    # Success check
    if max_diff < 1e-5:
        print("✅ SUCCESS: Models produce nearly identical outputs!")
        return True
    elif max_diff < 1e-3:
        print("✅ GOOD: Models produce similar outputs with small differences.")
        return True
    else:
        print("❌ FAILED: Models have significant differences.")
        return False

def test_file_save_load():
    """Test the file save/load approach with proper model building"""
    print("\n" + "=" * 60)
    print("TESTING FILE SAVE/LOAD APPROACH")
    print("=" * 60)
    
    # Create models
    keras_model, scratch_model, config = create_simple_models()
    
    # Build scratch model first
    dummy_input = np.random.randint(0, config['vocab_size'], (1, config['max_length']))
    _ = scratch_model.forward(dummy_input)
    
    # Create test data
    test_input = np.random.randint(0, config['vocab_size'], (2, config['max_length']))
    
    # Extract and save weights using the FIXED function
    print("1. Extracting and saving weights...")
    weights_path = 'test_weights.npz'
    weights_dict = extract_and_save_weights_fixed(keras_model, weights_path)
    
    # Check if file was actually created
    if not os.path.exists(weights_path):
        print("❌ ERROR: Weights file was not created!")
        return False
    
    # Load using the RNN model's load_weights method
    print("\n2. Loading weights from file...")
    try:
        scratch_model.load_weights(weights_path)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ ERROR loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test predictions
    print("\n3. Testing predictions...")
    keras_pred = keras_model.predict(test_input, verbose=0)
    scratch_pred = scratch_model.forward(test_input)
    
    max_diff = np.max(np.abs(keras_pred - scratch_pred))
    mean_diff = np.mean(np.abs(keras_pred - scratch_pred))
    
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    # Show sample predictions
    print(f"Sample predictions:")
    for i in range(min(2, len(test_input))):
        print(f"Sample {i+1}:")
        print(f"  Keras:    {keras_pred[i]}")
        print(f"  Scratch:  {scratch_pred[i]}")
        print(f"  Diff:     {np.abs(keras_pred[i] - scratch_pred[i])}")
        print()
    
    # Clean up
    if os.path.exists(weights_path):
        os.remove(weights_path)
        print(f"Cleaned up {weights_path}")
    
    # Success check
    if max_diff < 1e-5:
        print("✅ SUCCESS: File save/load works perfectly!")
        return True
    elif max_diff < 1e-3:
        print("✅ GOOD: File save/load works with small differences.")
        return True
    else:
        print("❌ FAILED: File save/load has significant differences.")
        return False

def test_keras_model_build_methods():
    """Test different ways to build Keras models"""
    print("\n" + "=" * 60)
    print("TESTING KERAS MODEL BUILD METHODS")
    print("=" * 60)
    
    config = {
        'vocab_size': 100,
        'embedding_dim': 16,
        'rnn_units': 8,
        'num_classes': 3,
        'max_length': 20,
    }
    
    # Method 1: Build with input shape
    print("Method 1: Build with input shape")
    model1 = keras.Sequential([
        layers.Embedding(input_dim=config['vocab_size'], output_dim=config['embedding_dim'], name='embedding'),
        layers.SimpleRNN(config['rnn_units'], name='simple_rnn'),
        layers.Dense(config['num_classes'], activation='softmax', name='classification')
    ])
    model1.build(input_shape=(None, config['max_length']))
    print(f"  Model 1 weights: {len(model1.weights)}")
    
    # Method 2: Call with dummy data
    print("\nMethod 2: Call with dummy data")
    model2 = keras.Sequential([
        layers.Embedding(input_dim=config['vocab_size'], output_dim=config['embedding_dim'], name='embedding'),
        layers.SimpleRNN(config['rnn_units'], name='simple_rnn'),
        layers.Dense(config['num_classes'], activation='softmax', name='classification')
    ])
    dummy_input = np.random.randint(0, config['vocab_size'], (1, config['max_length']))
    _ = model2(dummy_input)
    print(f"  Model 2 weights: {len(model2.weights)}")
    
    # Method 3: Add Input layer explicitly
    print("\nMethod 3: Add Input layer explicitly")
    model3 = keras.Sequential([
        layers.Input(shape=(config['max_length'],)),
        layers.Embedding(input_dim=config['vocab_size'], output_dim=config['embedding_dim'], name='embedding'),
        layers.SimpleRNN(config['rnn_units'], name='simple_rnn'),
        layers.Dense(config['num_classes'], activation='softmax', name='classification')
    ])
    print(f"  Model 3 weights: {len(model3.weights)}")
    
    # Test all models have weights
    for i, model in enumerate([model1, model2, model3], 1):
        if len(model.weights) > 0:
            print(f"✅ Method {i} successfully created weights")
        else:
            print(f"❌ Method {i} failed to create weights")

if __name__ == "__main__":
    print("Testing Keras model building methods...")
    test_keras_model_build_methods()
    
    print("\n\nRunning direct weight loading test...")
    success1 = test_forward_propagation()
    
    print("\nRunning file save/load test...")
    success2 = test_file_save_load()
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Weight loading works correctly.")
        print("You can now run the full experiment script.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED!")
        print("Direct loading:", "✅" if success1 else "❌")
        print("File save/load:", "✅" if success2 else "❌")
        print("=" * 60)