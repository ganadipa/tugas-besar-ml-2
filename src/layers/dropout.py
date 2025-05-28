import numpy as np
from layers.base import Layer

class DropoutLayer(Layer):
    """Dropout layer (acts as identity during inference)"""
    
    def __init__(self, rate: float = 0.5, name: str = None):
        super().__init__(name)
        self.rate = rate
        self.training = True  # Flag to track training vs inference mode
        self.mask = None # Cache for dropout mask
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through dropout layer (inference mode - no dropout)
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor (same as input during inference)
        """
        if self.training:
            # During training, apply dropout
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            return inputs * self.mask
        else:
            # During inference, pass through unchanged
            return inputs
    
    def backward(self, dout):
        if self.training and self.mask is not None:
            # Apply the same mask used in forward pass
            return dout * self.mask
        else:
            # During inference, pass gradient through unchanged
            return dout
        
    def set_training(self, training: bool):
        """Set training mode"""
        self.training = training
    
    def set_weights(self, weights):
        raise NotImplementedError("Not applicable!")
    
    def get_weights(self):
        raise NotImplementedError("Not applicable!")