import numpy as np
from activation_functions.base import ActivationFunction

class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)