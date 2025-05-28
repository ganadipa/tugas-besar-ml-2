import numpy as np
from layers.base import Layer

class DropoutLayer(Layer):
    """Dropout layer (acts as identity during inference)"""
    
    def __init__(self, rate: float = 0.5, name: str = None):
        super().__init__(name)
        self.rate = rate
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through dropout layer (inference mode - no dropout)
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor (same as input during inference)
        """
        return inputs