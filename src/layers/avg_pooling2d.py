import numpy as np
from layers.base import Layer
from typing import Tuple, Dict

class AveragePooling2D(Layer):
    def __init__(self, pool_size=(2, 2), name: str = None):
        super().__init__(name)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.input_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation for AveragePooling2D layer"""
        self.input_cache = x
        batch_size, height, width, channels = x.shape
        
        # Calculate output dimensions
        out_height = height // self.pool_size[0]
        out_width = width // self.pool_size[1]
        
        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        # Perform average pooling
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
                        avg_val = np.mean(patch)
                        output[b, i, j, c] = avg_val
        
        return output
    
    def backward(self, dout):
        """Backward propagation for AveragePooling2D layer"""
        dx = np.zeros_like(self.input_cache)
        batch_size, out_height, out_width, channels = dout.shape
        
        # Calculate the number of elements in each pool
        pool_area = self.pool_size[0] * self.pool_size[1]
        
        # Distribute gradients evenly across each pooling region
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.pool_size[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.pool_size[1]
                        w_end = w_start + self.pool_size[1]
                        
                        # Distribute gradient evenly across the pooling region
                        dx[b, h_start:h_end, w_start:w_end, c] += dout[b, i, j, c] / pool_area
        
        return dx
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        raise NotImplementedError("Not applicable!")
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Not applicable!")