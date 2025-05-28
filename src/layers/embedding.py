import numpy as np
from typing import Tuple, Optional, Dict
from layers.base import Layer


class EmbeddingLayer(Layer):
    """Embedding layer for converting tokens to dense vectors"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, name: str = None):
        super().__init__(name)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize embedding matrix
        self.embedding_matrix = np.random.normal(
            0, 0.1, (vocab_size, embedding_dim)
        ).astype(np.float32)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through embedding layer
        
        Args:
            inputs: Token indices of shape (batch_size, sequence_length)
            
        Returns:
            Embedded vectors of shape (batch_size, sequence_length, embedding_dim)
        """
        # Handle potential out-of-bounds indices
        inputs_clipped = np.clip(inputs, 0, self.vocab_size - 1)
        return self.embedding_matrix[inputs_clipped]
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"embedding_matrix": self.embedding_matrix}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        if "embedding_matrix" in weights:
            self.embedding_matrix = weights["embedding_matrix"]