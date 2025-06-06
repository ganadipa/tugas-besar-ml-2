import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from layers.base import Layer
from layers.embedding import EmbeddingLayer
from layers.dense import DenseLayer
from layers.dropout import DropoutLayer
from layers.lstm import LSTMLayer, BidirectionalLSTMLayer


class LSTMModel:
    """LSTM model for text classification"""
    
    def __init__(self, name: str = "LSTMModel"):
        self.name = name
        self.layers: List[Layer] = []
        self.built = False
    
    def add(self, layer: Layer):
        """Add a layer to the model"""
        self.layers.append(layer)
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the model (initialize all layers)"""
        current_shape = input_shape
        
        for layer in self.layers:
            if hasattr(layer, 'build'):
                layer.build(current_shape)
            
            # Update shape for next layer (approximate)
            if isinstance(layer, EmbeddingLayer):
                current_shape = current_shape + (layer.embedding_dim,)
            elif isinstance(layer, LSTMLayer):
                if layer.return_sequences:
                    current_shape = current_shape[:-1] + (layer.hidden_size,)
                else:
                    current_shape = (current_shape[0], layer.hidden_size)
            elif isinstance(layer, BidirectionalLSTMLayer):
                if layer.return_sequences:
                    current_shape = current_shape[:-1] + (layer.hidden_size * 2,)
                else:
                    current_shape = (current_shape[0], layer.hidden_size * 2)
            elif isinstance(layer, DenseLayer):
                current_shape = current_shape[:-1] + (layer.units,)
        
        self.built = True
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the model
        
        Args:
            inputs: Input tensor
            
        Returns:
            Model output
        """
        if not self.built and self.layers:
            self.build(inputs.shape)
        
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions (ensures inference mode)"""
        self.set_training(False)
        return self.forward(inputs)
    
    def set_training(self, training: bool):
        """Set training mode for all layers"""
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
    
    def get_weights(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get all model weights"""
        weights = {}
        for i, layer in enumerate(self.layers):
            layer_name = layer.name or f"layer_{i}"
            layer_weights = layer.get_weights()
            if layer_weights:
                weights[layer_name] = layer_weights
        return weights
    
    def set_weights(self, weights: Dict[str, Dict[str, np.ndarray]]):
        """Set all model weights with improved debugging"""
        print(f"Setting weights for LSTM model layers...")
        print(f"Available weight keys: {list(weights.keys())}")
        
        layers_with_weights = []
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.name or f"layer_{i}"
            print(f"Processing layer: {layer_name} (type: {type(layer).__name__})")
            
            if layer_name in weights:
                print(f"  Found weights for {layer_name}: {list(weights[layer_name].keys())}")
                layer.set_weights(weights[layer_name])
                layers_with_weights.append(layer_name)
            else:
                print(f"  No weights found for {layer_name}")
        
        print(f"Successfully loaded weights for layers: {layers_with_weights}")
        
        # Verify weights were loaded by checking get_weights
        loaded_weights = self.get_weights()
        print(f"Model now has weights for: {list(loaded_weights.keys())}")
    
    def save_weights(self, filepath: str):
        """Save model weights to file"""
        weights = self.get_weights()
        np.savez(filepath, **{
            f"{layer_name}_{weight_name}": weight_value
            for layer_name, layer_weights in weights.items()
            for weight_name, weight_value in layer_weights.items()
        })
    
    def load_weights(self, filepath: str):
        """Load model weights from file"""
        data = np.load(filepath)
        
        print(f"Loading LSTM weights from {filepath}")
        print(f"Available keys in file: {list(data.keys())}")
        
        # Reconstruct weights dictionary
        weights = {}
        
        for key, value in data.items():
            print(f"Processing key: {key}")
            
            # Handle embedding weights
            if key == 'embedding_embedding_matrix':
                if 'embedding' not in weights:
                    weights['embedding'] = {}
                weights['embedding']['embedding_matrix'] = value
                print(f"  -> embedding.embedding_matrix")
            
            # Handle LSTM layer weights
            elif key.startswith('lstm_0_'):
                weight_name = key[7:]  # Remove 'lstm_0_'
                if 'lstm_0' not in weights:
                    weights['lstm_0'] = {}
                weights['lstm_0'][weight_name] = value
                print(f"  -> lstm_0.{weight_name}")
            
            elif key.startswith('lstm_1_'):
                weight_name = key[7:]  # Remove 'lstm_1_'
                if 'lstm_1' not in weights:
                    weights['lstm_1'] = {}
                weights['lstm_1'][weight_name] = value
                print(f"  -> lstm_1.{weight_name}")
            
            elif key.startswith('lstm_2_'):
                weight_name = key[7:]  # Remove 'lstm_2_'
                if 'lstm_2' not in weights:
                    weights['lstm_2'] = {}
                weights['lstm_2'][weight_name] = value
                print(f"  -> lstm_2.{weight_name}")
            
            # Handle bidirectional LSTM weights
            elif key.startswith('bidirectional_lstm_0_'):
                weight_name = key[21:]  # Remove 'bidirectional_lstm_0_'
                if 'bidirectional_lstm_0' not in weights:
                    weights['bidirectional_lstm_0'] = {}
                weights['bidirectional_lstm_0'][weight_name] = value
                print(f"  -> bidirectional_lstm_0.{weight_name}")
            
            # Handle classification layer weights
            elif key == 'classification_W':
                if 'classification' not in weights:
                    weights['classification'] = {}
                weights['classification']['W'] = value
                print(f"  -> classification.W")
            
            elif key == 'classification_b':
                if 'classification' not in weights:
                    weights['classification'] = {}
                weights['classification']['b'] = value
                print(f"  -> classification.b")
            
            else:
                print(f"  Warning: Unknown key pattern: {key}")
        
        print(f"Final weights structure: {list(weights.keys())}")
        for layer_name, layer_weights in weights.items():
            print(f"  {layer_name}: {list(layer_weights.keys())}")
        
        self.set_weights(weights)
    
    def summary(self) -> str:
        """Generate a summary of the model architecture"""
        summary_lines = [f"Model: {self.name}"]
        summary_lines.append("=" * 50)
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.name or f"layer_{i}"
            layer_type = type(layer).__name__
            
            # Get layer parameters
            params = []
            if isinstance(layer, EmbeddingLayer):
                params.append(f"vocab_size={layer.vocab_size}")
                params.append(f"embedding_dim={layer.embedding_dim}")
            elif isinstance(layer, LSTMLayer):
                params.append(f"hidden_size={layer.hidden_size}")
                params.append(f"return_sequences={layer.return_sequences}")
            elif isinstance(layer, BidirectionalLSTMLayer):
                params.append(f"hidden_size={layer.hidden_size}")
                params.append(f"return_sequences={layer.return_sequences}")
            elif isinstance(layer, DenseLayer):
                params.append(f"units={layer.units}")
                if layer.activation_name:
                    params.append(f"activation={layer.activation_name}")
            elif isinstance(layer, DropoutLayer):
                params.append(f"rate={layer.rate}")
            
            param_str = ", ".join(params)
            summary_lines.append(f"{layer_name} ({layer_type}): {param_str}")
        
        summary_lines.append("=" * 50)
        return "\n".join(summary_lines)


class LSTMModelBuilder:
    """Builder class for creating LSTM models"""
    
    @staticmethod
    def create_lstm_model(
        vocab_size: int,
        embedding_dim: int,
        lstm_units: int,
        num_classes: int,
        num_lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout_rate: float = 0.2
    ) -> LSTMModel:
        """
        Create an LSTM model for text classification
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units
            num_classes: Number of output classes
            num_lstm_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout rate
            
        Returns:
            Configured LSTM model
        """
        model = LSTMModel(name="LSTM")
        
        # Embedding layer
        model.add(EmbeddingLayer(vocab_size, embedding_dim, name="embedding"))
        
        # LSTM layers
        for i in range(num_lstm_layers):
            return_sequences = i < num_lstm_layers - 1  # Only last layer doesn't return sequences
            
            if bidirectional:
                model.add(BidirectionalLSTMLayer(
                    hidden_size=lstm_units,
                    return_sequences=return_sequences,
                    name=f"bidirectional_lstm_{i}"
                ))
            else:
                model.add(LSTMLayer(
                    hidden_size=lstm_units,
                    return_sequences=return_sequences,
                    name=f"lstm_{i}"
                ))
            
            # Add dropout after each LSTM layer except the last
            if i < num_lstm_layers - 1:
                model.add(DropoutLayer(rate=dropout_rate, name=f"dropout_{i}"))
        
        # Final dropout before classification
        model.add(DropoutLayer(rate=dropout_rate, name="dropout_final"))
        
        # Classification layer
        model.add(DenseLayer(num_classes, activation='softmax', name="classification"))
        
        return model
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> LSTMModel:
        """
        Create LSTM model from configuration dictionary
        
        Args:
            config: Model configuration
            
        Returns:
            Configured LSTM model
        """
        return LSTMModelBuilder.create_lstm_model(**config)