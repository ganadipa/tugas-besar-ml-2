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

        # Training parameters
        self.learning_rate = 0.001
        self.beta1 = 0.9  # Adam parameter
        self.beta2 = 0.999  # Adam parameter
        self.epsilon = 1e-8  # Adam parameter
        self.t = 0  # Time step for Adam
        
        # Initialize Adam optimizer parameters
        self.m_weights = []  # First moment estimates for weights
        self.v_weights = []  # Second moment estimates for weights
        self.m_biases = []   # First moment estimates for biases
        self.v_biases = []   # Second moment estimates for biases
        
    
    def build_model(self):
        """Build the same architecture as the Keras model"""
        # First Conv2D block
        pass

    def load_weights_from_keras(self, npz_file_path):
        """Load weights and architecture from .npz file"""
        print(f"Loading weights and architecture from {npz_file_path}...")
        
        # Load the .npz file
        data = np.load(npz_file_path, allow_pickle=True)
        
        # Extract architecture and reconstruct Keras model
        architecture = data['architecture'].item()
        keras_model = keras.models.model_from_json(architecture)
        
        # Load weights back into the Keras model
        weights = [data[f'weight_{i}'] for i in range(len([k for k in data.keys() if k.startswith('weight_')]))]
        keras_model.set_weights(weights)
        
        # Parse the architecture and build our custom layers
        import json
        config = json.loads(architecture)
        
        # Clear existing layers
        self.layers = []
        
        # Build layers from config
        for layer_config in config['config']['layers']:
            layer_type = layer_config['class_name']
            layer_params = layer_config['config']
            
            if layer_type == 'Conv2D':
                filters = layer_params['filters']
                kernel_size = tuple(layer_params['kernel_size'])
                activation = layer_params['activation']
                input_shape = layer_params.get('batch_input_shape')
                if input_shape:
                    input_shape = tuple(input_shape[1:])  # Remove batch dimension
                
                self.layers.append(Conv2D(filters, kernel_size, activation, input_shape, name="Conv2D"))
                print(f"Added Conv2D layer: filters={filters}, kernel_size={kernel_size}, activation={activation}")
                
            elif layer_type == 'MaxPooling2D':
                pool_size = tuple(layer_params['pool_size'])
                self.layers.append(MaxPooling2D(pool_size, name="MaxPooling2D"))
                print(f"Added MaxPooling2D layer: pool_size={pool_size}")
                
            elif layer_type == 'Flatten':
                self.layers.append(Flatten(name="Flatten"))
                print("Added Flatten layer")
                
            elif layer_type == 'Dense':
                units = layer_params['units']
                activation = layer_params['activation']
                self.layers.append(DenseLayer(units, activation, name="DenseLayer"))
                print(f"Added Dense layer: units={units}, activation={activation}")
                
            elif layer_type == 'Dropout':
                # Skip dropout layers as they don't affect inference
                print("Skipped Dropout layer (not needed for inference)")
                continue
                
            else:
                print(f"Warning: Unknown layer type {layer_type}, skipping...")
        
        # Get the weights from the reconstructed model
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
        
        print("All weights and architecture loaded successfully!")
        self._initialize_adam_parameters()

    def _initialize_adam_parameters(self):
        """Initialize Adam optimizer parameters"""
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                self.m_weights.append(np.zeros_like(layer.W.T))
                self.v_weights.append(np.zeros_like(layer.W.T))
                self.m_biases.append(np.zeros_like(layer.b.T))
                self.v_biases.append(np.zeros_like(layer.b.T))
            elif isinstance(layer, Conv2D):
                self.m_weights.append(np.zeros_like(layer.weights))
                self.v_weights.append(np.zeros_like(layer.weights))
                self.m_biases.append(np.zeros_like(layer.biases))
                self.v_biases.append(np.zeros_like(layer.biases))
    
    # def load_weights_from_keras(self, h5_file_path):
    #     """Load weights from Keras .h5 file"""
    #     print(f"Loading weights from {h5_file_path}...")
        
    #     # Load the Keras model
    #     keras_model = keras.models.load_model(h5_file_path)
    #     keras_weights = keras_model.get_weights()
        
    #     # Map weights to our layers
    #     weight_idx = 0
    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, Conv2D):
    #             # Conv2D layers have weights and biases
    #             layer.set_weights({"W": keras_weights[weight_idx], "b": keras_weights[weight_idx + 1]})
    #             weight_idx += 2
    #             print(f"Loaded weights for Conv2D layer {i}")
    #         elif isinstance(layer, DenseLayer):
    #             # Dense layers have weights and biases
    #             layer.set_weights({"W": keras_weights[weight_idx].T, "b": keras_weights[weight_idx + 1].T})
    #             weight_idx += 2
    #             print(f"Loaded weights for Dense layer {i}")
        
    #     print("All weights loaded successfully!")
    
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
    
    def compute_loss_and_gradients(self, predictions, targets):
        """Compute loss and initial gradient for backpropagation"""
        batch_size = predictions.shape[0]
        
        # Convert targets to integers if needed
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.flatten()
        
        # Compute cross-entropy loss
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        loss = -np.log(predictions_clipped[np.arange(batch_size), targets])
        avg_loss = np.mean(loss)
        
        # Compute gradient of loss w.r.t. predictions (for softmax + cross-entropy)
        dout = predictions.copy()
        dout[np.arange(batch_size), targets] -= 1
        dout /= batch_size
        
        return avg_loss, dout
    
    def update_weights_adam(self, gradients):
        """Update weights using Adam optimizer"""
        self.t += 1  # Increment time step
        
        param_idx = 0
        for layer in self.layers:
            if isinstance(layer, (Conv2D, DenseLayer)):
                dw, db = gradients[param_idx]
                
                # Update weights
                self.m_weights[param_idx] = (self.beta1 * self.m_weights[param_idx] + 
                                           (1 - self.beta1) * dw)
                self.v_weights[param_idx] = (self.beta2 * self.v_weights[param_idx] + 
                                           (1 - self.beta2) * (dw ** 2))
                
                # Bias correction
                m_corrected = self.m_weights[param_idx] / (1 - self.beta1 ** self.t)
                v_corrected = self.v_weights[param_idx] / (1 - self.beta2 ** self.t)
                
                # Update weights
                if isinstance(layer, Conv2D):
                    layer.weights -= (self.learning_rate * m_corrected / 
                                (np.sqrt(v_corrected) + self.epsilon))
                else:
                    layer.W -= ((self.learning_rate * m_corrected / 
                                (np.sqrt(v_corrected) + self.epsilon))).T
                
                # Update biases
                self.m_biases[param_idx] = (self.beta1 * self.m_biases[param_idx] + 
                                          (1 - self.beta1) * db)
                self.v_biases[param_idx] = (self.beta2 * self.v_biases[param_idx] + 
                                          (1 - self.beta2) * (db ** 2))
                
                # Bias correction for biases
                m_corrected_b = self.m_biases[param_idx] / (1 - self.beta1 ** self.t)
                v_corrected_b = self.v_biases[param_idx] / (1 - self.beta2 ** self.t)
                
                # Update biases
                if isinstance(layer, Conv2D):
                    layer.biases -= (self.learning_rate * m_corrected_b / 
                               (np.sqrt(v_corrected_b) + self.epsilon))
                else:
                    layer.b -= ((self.learning_rate * m_corrected_b / 
                                (np.sqrt(v_corrected_b) + self.epsilon))).T
                
                param_idx += 1
    
    def train_batch(self, x_batch, y_batch):
        """Train on a single batch"""
        # Forward pass
        predictions = self.forward(x_batch)
        
        # Compute loss and gradients
        loss, dout = self.compute_loss_and_gradients(predictions, y_batch)
        
        # Backward pass
        gradients = self.backward(dout)
        
        # Update weights
        self.update_weights_adam(gradients)
        
        # Compute accuracy
        accuracy = self.compute_accuracy(predictions, y_batch)
        
        return loss, accuracy
    
    def validate(self, x_val, y_val, batch_size=32):
        """Validate the model"""
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for i in range(0, len(x_val), batch_size):
            x_batch = x_val[i:i+batch_size]
            y_batch = y_val[i:i+batch_size]
            
            # Forward pass only
            predictions = self.forward(x_batch)
            loss = self.compute_loss(predictions, y_batch)
            accuracy = self.compute_accuracy(predictions, y_batch)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
        
        return total_loss / num_batches, total_accuracy / num_batches
    
    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=10, batch_size=32, verbose=1):
        """Train the model"""
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        print(f"Training for {epochs} epochs...")
        print(f"Training samples: {len(x_train)}, Batch size: {batch_size}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Training loop
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Train on batch
                batch_loss, batch_accuracy = self.train_batch(x_batch, y_batch)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                num_batches += 1
                
                # Print progress
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {num_batches}: "
                          f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
            
            # Calculate average metrics
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_accuracy)
            
            # Validation
            if x_val is not None and y_val is not None:
                val_loss, val_accuracy = self.validate(x_val, y_val, batch_size)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
            
            # Learning rate decay
            if (epoch + 1) % 10 == 0:
                self.learning_rate *= 0.9
                print(f"Learning rate reduced to: {self.learning_rate:.6f}")
        
        return history
    
    def save_weights(self, filepath):
        """Save model weights to a file"""
        weights_data = []
        for layer in self.layers:
            if isinstance(layer, (Conv2D, DenseLayer)):
                weights_data.append({
                    'weights': layer.weights,
                    'biases': layer.biases,
                    'layer_type': type(layer).__name__
                })
        
        np.save(filepath, weights_data)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """Load model weights from a file"""
        weights_data = np.load(filepath, allow_pickle=True)
        
        param_idx = 0
        for layer in self.layers:
            if isinstance(layer, (Conv2D, DenseLayer)):
                layer.weights = weights_data[param_idx]['weights']
                layer.biases = weights_data[param_idx]['biases']
                param_idx += 1
        
        print(f"Weights loaded from {filepath}")

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
    model.load_weights_from_keras('results/cifar10_cnn_model.npz')
    
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
    # keras_model = keras.models.load_model('results/cifar10_cnn_model.npz')

    # Load the .npz file and reconstruct Keras model
    data = np.load('results/cifar10_cnn_model.npz', allow_pickle=True)
    architecture = data['architecture'].item()
    keras_model = keras.models.model_from_json(architecture)
    weights = [data[f'weight_{i}'] for i in range(len([k for k in data.keys() if k.startswith('weight_')]))]
    keras_model.set_weights(weights)
    
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
    our_model.load_weights_from_keras('results/cifar10_cnn_model.npz')
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

def load_and_preprocess_data():
    """Load CIFAR-10 data and preprocess it"""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Create validation split (4:1 ratio as requested)
    # Split training data: 80% train, 20% validation
    split_idx = int(0.8 * len(x_train))
    
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Validation set: {x_val.shape[0]} samples")
    print(f"Test set: {x_test.shape[0]} samples")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_from_scratch():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    x_train = x_train[:100]
    y_train = y_train[:100]

    x_val = x_val[:100]
    y_val = y_val[:100]

    x_test = x_test[:100]
    y_test = y_test[:100]

    model = CNNFromScratch()
    model.load_weights_from_keras('results/cifar10_cnn_model.npz')

    history = model.fit(x_train, y_train, x_val, y_val, epochs=10)

    plot_training_history(history)

    # Evaluate performance
    test_loss, test_accuracy = model.validate(x_test, y_test)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save trained weights
    model.save_weights('my_trained_model.npy')

if __name__ == "__main__":
    # Test the implementation
    # model = test_model()

    train_from_scratch()

    # Compare with Keras
    # compare_with_keras()
    
    print("\nCNN from scratch implementation completed!")
    print("The model successfully loads weights from the Keras .h5 file")
    print("and performs inference with the same architecture.")