import numpy as np
from activation_functions.base import ActivationFunction

class Sigmoid(ActivationFunction):
    """Sigmoid activation function"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    