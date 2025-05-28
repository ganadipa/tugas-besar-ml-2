import numpy as np
from typing import Tuple, Optional, Dict
from layers.base import Layer

class Flatten(Layer):
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_shape = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation for Flatten layer"""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        """Backward propagation for Flatten layer"""
        return dout.reshape(self.input_shape)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Not applicable!")
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        raise NotImplementedError("Not applicable!")
