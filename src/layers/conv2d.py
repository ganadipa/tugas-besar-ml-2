import numpy as np
from layers.base import Layer
from typing import Tuple, Dict

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, activation='relu', input_shape=None, name: str = None):
        super().__init__(name)

        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.activation = activation
        self.input_shape = input_shape
        
        # Initialize weights and biases (will be loaded from file)
        self.weights = None
        self.biases = None
        
        # Cache for backward propagation
        self.input_cache = None
        self.output_cache = None
        
    def initialize_weights(self, input_channels):
        """Initialize weights using He initialization"""
        fan_in = self.kernel_size[0] * self.kernel_size[1] * input_channels
        std = np.sqrt(2.0 / fan_in)
        self.weights = np.random.normal(0, std, 
                                      (self.kernel_size[0], self.kernel_size[1], 
                                       input_channels, self.filters))
        self.biases = np.zeros(self.filters)
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights from loaded model"""
        self.weights = weights["W"]
        self.biases = weights["b"]

    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.weights is None:
            return {}
        return {"W": self.weights, "b": self.biases}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation for Conv2D layer"""
        self.input_cache = x
        batch_size, height, width, channels = x.shape
        
        # Calculate output dimensions
        out_height = height - self.kernel_size[0] + 1
        out_width = width - self.kernel_size[1] + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, self.filters))
        
        # Perform convolution
        for b in range(batch_size):
            for f in range(self.filters):
                for i in range(out_height):
                    for j in range(out_width):
                        # Extract patch
                        patch = x[b, i:i+self.kernel_size[0], j:j+self.kernel_size[1], :]
                        # Convolution operation
                        output[b, i, j, f] = np.sum(patch * self.weights[:, :, :, f]) + self.biases[f]
        
        # Apply activation
        if self.activation == 'relu':
            output = np.maximum(0, output)
        
        self.output_cache = output
        return output
    
    def backward(self, dout):
        """Backward propagation for Conv2D layer"""
        x = self.input_cache
        batch_size, height, width, channels = x.shape
        
        # Initialize gradients
        dx = np.zeros_like(x)
        dw = np.zeros_like(self.weights)
        db = np.sum(dout, axis=(0, 1, 2))
        
        # Apply activation derivative
        if self.activation == 'relu':
            dout = dout * (self.output_cache > 0)
        
        # Calculate gradients
        for b in range(batch_size):
            for f in range(self.filters):
                for i in range(dout.shape[1]):
                    for j in range(dout.shape[2]):
                        # Gradient w.r.t weights
                        patch = x[b, i:i+self.kernel_size[0], j:j+self.kernel_size[1], :]
                        dw[:, :, :, f] += patch * dout[b, i, j, f]
                        
                        # Gradient w.r.t input
                        dx[b, i:i+self.kernel_size[0], j:j+self.kernel_size[1], :] += (
                            self.weights[:, :, :, f] * dout[b, i, j, f]
                        )
        
        return dx, dw, db