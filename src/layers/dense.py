import numpy as np
from typing import Tuple, Dict
from layers.base import Layer
from activation_functions.base import ActivationFunction
from activation_functions.tanh import Tanh
from activation_functions.sigmoid import Sigmoid
from activation_functions.relu import ReLU
from activation_functions.softmax import Softmax

class DenseLayer(Layer):
    """Dense (fully connected) layer"""
    
    def __init__(self, units: int, activation: str = None, name: str = None):
        super().__init__(name)
        self.units = units
        self.activation_name = activation
        self.activation = self._get_activation(activation) if activation else None

        # Cache for backward propagation
        self.input_cache = None
        self.output_cache = None
        
        # Weights will be initialized in build()
        self.W = None
        self.b = None
    
    def _get_activation(self, activation: str) -> ActivationFunction:
        """Get activation function by name"""
        activations = {
            'tanh': Tanh,
            'sigmoid': Sigmoid,
            'relu': ReLU,
            'softmax': Softmax
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation.lower()]
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer based on input shape"""
        input_size = input_shape[-1]
        
        # Initialize weights
        self.W = np.random.normal(0, 0.1, (self.units, input_size)).astype(np.float32)
        self.b = np.zeros((self.units,), dtype=np.float32)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through dense layer
        
        Args:
            inputs: Input tensor, shape (batch_size, input_size)
            
        Returns:
            Output tensor, shape (batch_size, units)
        """
        self.input_cache = inputs
        if self.W is None:
            self.build(inputs.shape)
        
        # Linear transformation
        output = np.dot(inputs, self.W.T) + self.b
        
        # Apply activation if specified
        if self.activation is not None:
            output = self.activation.forward(output)

        self.output_cache = output
        
        return output
    
    def backward(self, dout):
        x = self.input_cache

        if self.activation is not None:
            dout = self.activation.backward(dout, self.output_cache)

        dx = np.dot(dout, self.W)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)

        return dx, dw, db
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.W is None:
            return {}
        return {"W": self.W, "b": self.b}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        if "W" in weights:
            self.W = weights["W"]
        if "b" in weights:
            self.b = weights["b"]