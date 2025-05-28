from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Any, Dict


class Layer(ABC):
    """Abstract base class for all neural network layers"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.trainable = True
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward propagation through the layer"""
        pass

    # Dont make this backward just yet
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward propagation through the layer"""
        pass
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get layer weights"""
        return {}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set layer weights"""
        pass