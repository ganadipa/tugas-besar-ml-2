import numpy as np
from activation_functions.base import ActivationFunction



class ReLU(ActivationFunction):
    """ReLU activation function"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)