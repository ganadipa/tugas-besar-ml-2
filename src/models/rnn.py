import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from layers.base import Layer
from layers.embedding import EmbeddingLayer
from layers.bidirectional import BidirectionalLayer
from layers.simple_rnn import SimpleRNNLayer
from layers.dense import DenseLayer
from layers.dropout import DropoutLayer


class RNNModel:
    """Simple RNN model for text classification"""
    
    def __init__(self, name: str = "RNNModel"):
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
            elif isinstance(layer, SimpleRNNLayer):
                if layer.return_sequences:
                    current_shape = current_shape[:-1] + (layer.hidden_size,)
                else:
                    current_shape = (current_shape[0], layer.hidden_size)
            elif isinstance(layer, BidirectionalLayer):
                if hasattr(layer.forward_layer, 'return_sequences'):
                    if layer.forward_layer.return_sequences:
                        current_shape = current_shape[:-1] + (layer.forward_layer.hidden_size * 2,)
                    else:
                        current_shape = (current_shape[0], layer.forward_layer.hidden_size * 2)
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
        """Make predictions (alias for forward)"""
        return self.forward(inputs)
    
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
        """Set all model weights"""
        for i, layer in enumerate(self.layers):
            layer_name = layer.name or f"layer_{i}"
            if layer_name in weights:
                layer.set_weights(weights[layer_name])
    
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
        
        # Reconstruct weights dictionary
        weights = {}
        for key, value in data.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                layer_name, weight_name = parts
                if layer_name not in weights:
                    weights[layer_name] = {}
                weights[layer_name][weight_name] = value
        
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
            elif isinstance(layer, SimpleRNNLayer):
                params.append(f"hidden_size={layer.hidden_size}")
                params.append(f"activation={layer.activation}")
                params.append(f"return_sequences={layer.return_sequences}")
            elif isinstance(layer, BidirectionalLayer):
                params.append(f"hidden_size={layer.forward_layer.hidden_size}")
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


class RNNModelBuilder:
    """Builder class for creating RNN models"""
    
    @staticmethod
    def create_simple_rnn_model(
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int,
        num_classes: int,
        num_rnn_layers: int = 1,
        bidirectional: bool = False,
        dropout_rate: float = 0.2,
        activation: str = 'tanh'
    ) -> RNNModel:
        """
        Create a simple RNN model for text classification
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            rnn_units: Number of RNN units
            num_classes: Number of output classes
            num_rnn_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            dropout_rate: Dropout rate
            activation: RNN activation function
            
        Returns:
            Configured RNN model
        """
        model = RNNModel(name="SimpleRNN")
        
        # Embedding layer
        model.add(EmbeddingLayer(vocab_size, embedding_dim, name="embedding"))
        
        # RNN layers
        for i in range(num_rnn_layers):
            return_sequences = i < num_rnn_layers - 1  # Only last layer doesn't return sequences
            
            rnn_layer = SimpleRNNLayer(
                hidden_size=rnn_units,
                activation=activation,
                return_sequences=return_sequences,
                name=f"rnn_{i}"
            )
            
            if bidirectional:
                model.add(BidirectionalLayer(
                    SimpleRNNLayer,
                    hidden_size=rnn_units,
                    activation=activation,
                    return_sequences=return_sequences,
                    name=f"bidirectional_rnn_{i}"
                ))
            else:
                model.add(rnn_layer)
            
            # Add dropout after each RNN layer except the last
            if i < num_rnn_layers - 1:
                model.add(DropoutLayer(rate=dropout_rate, name=f"dropout_{i}"))
        
        # Final dropout before classification
        model.add(DropoutLayer(rate=dropout_rate, name="dropout_final"))
        
        # Classification layer
        model.add(DenseLayer(num_classes, activation='softmax', name="classification"))
        
        return model
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> RNNModel:
        """
        Create RNN model from configuration dictionary
        
        Args:
            config: Model configuration
            
        Returns:
            Configured RNN model
        """
        return RNNModelBuilder.create_simple_rnn_model(**config)