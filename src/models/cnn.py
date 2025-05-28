import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
from typing import List
from layers.dense import DenseLayer
from layers.flatten import Flatten
from layers.max_pooling2d import MaxPooling2D
from layers.conv2d import Conv2D
from layers.base import Layer
import keras

class CNNFromScratch:
    def __init__(self):
        self.layers: List[Layer] = []
        self.build_model()
    
    def build_model(self):
        """Build the same architecture as the Keras model"""
        # First Conv2D block
        self.layers.append(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name="Conv2D"))
        self.layers.append(MaxPooling2D((2, 2), name="MaxPooling2D"))
        
        # Second Conv2D block
        self.layers.append(Conv2D(64, (3, 3), activation='relu', name="Conv2D"))
        self.layers.append(MaxPooling2D((2, 2), name="MaxPooling2D"))
        
        # Third Conv2D block
        self.layers.append(Conv2D(128, (3, 3), activation='relu', name="Conv2D"))
        self.layers.append(MaxPooling2D((2, 2), name="MaxPooling2D"))
        
        # Flatten layer
        self.layers.append(Flatten(name="Flatten"))
        
        # Dense layers
        self.layers.append(DenseLayer(128, activation='relu', name="Dense"))
        self.layers.append(DenseLayer(64, activation='relu', name="Dense"))
        self.layers.append(DenseLayer(10, activation='softmax', name="Dense"))
    
    def load_weights_from_keras(self, h5_file_path):
        """Load weights from Keras .h5 file"""
        print(f"Loading weights from {h5_file_path}...")
        
        # Load the Keras model
        keras_model = keras.models.load_model(h5_file_path)
        keras_weights = keras_model.get_weights()
        
        # Map weights to our layers
        weight_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv2D):
                # Conv2D layers have weights and biases
                layer.set_weights({"W": keras_weights[weight_idx], "b": keras_weights[weight_idx + 1]})
                weight_idx += 2
                print(f"Loaded weights for Conv2D layer {i}")
            elif isinstance(layer, DenseLayer):
                # Dense layers have weights and biases
                layer.set_weights({"W": keras_weights[weight_idx].T, "b": keras_weights[weight_idx + 1].T})
                weight_idx += 2
                print(f"Loaded weights for Dense layer {i}")
        
        print("All weights loaded successfully!")
    
    def forward(self, x):
        """Forward propagation through the entire network"""
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            print(f"After layer {i} ({type(layer).__name__}): {x.shape}")
        return x
    
    def backward(self, dout):
        """Backward propagation through the entire network"""
        gradients = []
        
        # Reverse iteration through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if isinstance(layer, (Conv2D, DenseLayer)):
                dx, dw, db = layer.backward(dout)
                gradients.append((dw, db))
                dout = dx
            else:
                dout = layer.backward(dout)
        
        return list(reversed(gradients))
    
    def predict(self, x):
        """Make predictions"""
        return self.forward(x)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray):
        """Compute sparse categorical crossentropy loss"""
        batch_size = predictions.shape[0]
        # Convert targets to one-hot if needed
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.flatten()
        
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        
        # Compute cross-entropy loss
        loss = -np.log(predictions[np.arange(batch_size), targets])
        return np.mean(loss)
    
    def compute_accuracy(self, predictions, targets):
        """Compute accuracy"""
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.flatten()
        
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == targets)

def test_model():
    """Test the from-scratch CNN with CIFAR-10 data"""
    # Load CIFAR-10 test data
    print("Loading CIFAR-10 test data...")
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    
    # Take a small subset for testing
    test_samples = 100
    x_test_small = x_test[:test_samples]
    y_test_small = y_test[:test_samples]
    
    # Create and load model
    print("Creating CNN from scratch...")
    model = CNNFromScratch()
    
    # Load weights from the saved Keras model
    model.load_weights_from_keras('results/cifar10_cnn_model.h5')
    
    # Make predictions
    print(f"\nTesting on {test_samples} samples...")
    predictions = model.predict(x_test_small)
    
    # Compute metrics
    loss = model.compute_loss(predictions, y_test_small)
    accuracy = model.compute_accuracy(predictions, y_test_small)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Show some predictions
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\nSample Predictions:")
    for i in range(5):
        predicted_class = np.argmax(predictions[i])
        true_class = y_test_small[i][0]
        confidence = predictions[i][predicted_class]
        
        print(f"Sample {i+1}: Predicted={class_names[predicted_class]} "
              f"(confidence: {confidence:.3f}), True={class_names[true_class]}")
    
    return model

def compare_with_keras():
    """Compare our implementation with the original Keras model"""
    print("Comparing with original Keras model...")
    
    # Load Keras model
    keras_model = keras.models.load_model('results/cifar10_cnn_model.h5')
    
    # Load test data
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    
    # Take small subset
    test_samples = 100
    x_test_small = x_test[:test_samples]
    y_test_small = y_test[:test_samples]
    
    # Get Keras predictions
    keras_predictions = keras_model.predict(x_test_small, verbose=0)
    
    # Get our model predictions
    our_model = CNNFromScratch()
    our_model.load_weights_from_keras('results/cifar10_cnn_model.h5')
    our_predictions = our_model.predict(x_test_small)
    
    # Compare predictions
    print("\nPrediction Comparison (first 3 samples):")
    for i in range(3):
        print(f"\nSample {i+1}:")
        print(f"Keras prediction: {keras_predictions[i][:5]}...")  # First 5 values
        print(f"Our prediction:   {our_predictions[i][:5]}...")    # First 5 values
        print(f"Difference:       {np.abs(keras_predictions[i] - our_predictions[i])[:5]}...")
    
    # Overall difference
    max_diff = np.max(np.abs(keras_predictions - our_predictions))
    mean_diff = np.mean(np.abs(keras_predictions - our_predictions))
    
    print(f"\nOverall comparison:")
    print(f"Maximum difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

if __name__ == "__main__":
    import sys
    print(sys.path)
    # Test the implementation
    model = test_model()
    
    # Compare with Keras
    compare_with_keras()
    
    print("\nCNN from scratch implementation completed!")
    print("The model successfully loads weights from the Keras .h5 file")
    print("and performs inference with the same architecture.")