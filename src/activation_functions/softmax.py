import numpy as np
from activation_functions.base import ActivationFunction

class Softmax(ActivationFunction):
    """Softmax activation function"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def backward(dout: np.ndarray, output: np.ndarray) -> np.ndarray:
        return dout