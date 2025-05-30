# src/layers/bidirectional.py - FIXED VERSION

import numpy as np
from typing import Dict
from layers.base import Layer


class BidirectionalLayer(Layer):
    """Bidirectional RNN wrapper"""
    
    def __init__(self, layer_class, **layer_kwargs):
        name = layer_kwargs.pop('name', None)
        super().__init__(name)
        
        # Create forward and backward layers
        self.forward_layer = layer_class(**layer_kwargs, name=f"{self.name}_forward")
        self.backward_layer = layer_class(**layer_kwargs, name=f"{self.name}_backward")
        
        # Store kwargs for potential reconstruction
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs
        
        # Store pending weights for loading before build
        self._pending_weights = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through bidirectional layer
        
        Args:
            inputs: Input sequences, shape (batch_size, sequence_length, input_size)
            
        Returns:
            Concatenated forward and backward outputs
        """
        # Forward direction
        forward_output = self.forward_layer.forward(inputs)
        
        # Backward direction (reverse the sequence)
        backward_input = inputs[:, ::-1, :]  # Reverse time dimension
        backward_output = self.backward_layer.forward(backward_input)
        
        if len(backward_output.shape) == 3:  # return_sequences=True
            # Reverse the backward output to match forward direction
            backward_output = backward_output[:, ::-1, :]
            # Concatenate along feature dimension
            return np.concatenate([forward_output, backward_output], axis=-1)
        else:  # return_sequences=False
            # Concatenate along feature dimension
            return np.concatenate([forward_output, backward_output], axis=-1)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        if self._pending_weights is not None:
            return self._pending_weights
            
        forward_weights = self.forward_layer.get_weights()
        backward_weights = self.backward_layer.get_weights()
        
        # Prefix weights with forward/backward
        weights = {}
        for k, v in forward_weights.items():
            weights[f"forward_{k}"] = v
        for k, v in backward_weights.items():
            weights[f"backward_{k}"] = v
        
        return weights
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights for both forward and backward layers"""
        print(f"Setting weights for bidirectional layer {self.name}: {list(weights.keys())}")
        
        forward_weights = {}
        backward_weights = {}
        
        for k, v in weights.items():
            if k.startswith("forward_"):
                forward_weights[k[8:]] = v  # Remove "forward_" prefix
            elif k.startswith("backward_"):
                backward_weights[k[9:]] = v  # Remove "backward_" prefix
        
        if forward_weights:
            print(f"Setting forward weights: {list(forward_weights.keys())}")
            self.forward_layer.set_weights(forward_weights)
            
        if backward_weights:
            print(f"Setting backward weights: {list(backward_weights.keys())}")
            self.backward_layer.set_weights(backward_weights)
            
        # Store pending weights if layers aren't built yet
        if not forward_weights and not backward_weights:
            self._pending_weights = weights.copy()