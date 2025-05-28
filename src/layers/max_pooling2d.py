import numpy as np
from layers.base import Layer
from typing import Tuple, Dict

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), name: str = None):
        super().__init__(name)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.input_cache = None
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation for MaxPooling2D layer"""
        self.input_cache = x
        batch_size, height, width, channels = x.shape
        
        # Calculate output dimensions
        out_height = height // self.pool_size[0]
        out_width = width // self.pool_size[1]
        
        # Initialize output and mask
        output = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = np.zeros_like(x)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Extract patch
                        h_start = i * self.pool_size[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.pool_size[1]
                        w_end = w_start + self.pool_size[1]
                        
                        patch = x[b, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(patch)
                        output[b, i, j, c] = max_val
                        
                        # Create mask for backward propagation
                        mask_patch = (patch == max_val)
                        self.mask[b, h_start:h_end, w_start:w_end, c] = mask_patch
        
        return output
    
    def backward(self, dout):
        """Backward propagation for MaxPooling2D layer"""
        dx = np.zeros_like(self.input_cache)
        batch_size, out_height, out_width, channels = dout.shape
        
        # Distribute gradients using mask
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.pool_size[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.pool_size[1]
                        w_end = w_start + self.pool_size[1]
                        
                        dx[b, h_start:h_end, w_start:w_end, c] += (
                            self.mask[b, h_start:h_end, w_start:w_end, c] * dout[b, i, j, c]
                        )
        
        return dx
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        raise NotImplementedError("Not applicable!")
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Not applicable!")
