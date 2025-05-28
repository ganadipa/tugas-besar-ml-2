import numpy as np
from typing import Dict, Tuple
from layers.base import Layer
from activation_functions.base import ActivationFunction
from activation_functions.tanh import Tanh
from activation_functions.sigmoid import Sigmoid
from activation_functions.relu import ReLU

class SimpleRNNCell:
    """Simple RNN cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, activation: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.W_ih = np.random.normal(0, 0.1, (hidden_size, input_size)).astype(np.float32)
        self.W_hh = np.random.normal(0, 0.1, (hidden_size, hidden_size)).astype(np.float32)
        self.b_h = np.zeros((hidden_size,), dtype=np.float32)
        
        # Set activation function
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str) -> ActivationFunction:
        """Get activation function by name"""
        activations = {
            'tanh': Tanh,
            'sigmoid': Sigmoid,
            'relu': ReLU
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        return activations[activation.lower()]
    
    def forward_step(self, input_t: np.ndarray, hidden_t_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for one time step
        
        Args:
            input_t: Input at time t, shape (batch_size, input_size)
            hidden_t_prev: Previous hidden state, shape (batch_size, hidden_size)
            
        Returns:
            New hidden state, shape (batch_size, hidden_size)
        """
        # h_t = activation(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
        linear_out = (
            np.dot(input_t, self.W_ih.T) + 
            np.dot(hidden_t_prev, self.W_hh.T) + 
            self.b_h
        )
        return self.activation.forward(linear_out)
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            "W_ih": self.W_ih,
            "W_hh": self.W_hh,
            "b_h": self.b_h
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        if "W_ih" in weights:
            self.W_ih = weights["W_ih"]
        if "W_hh" in weights:
            self.W_hh = weights["W_hh"]
        if "b_h" in weights:
            self.b_h = weights["b_h"]


class SimpleRNNLayer(Layer):
    """Simple RNN layer implementation"""
    
    def __init__(self, hidden_size: int, activation: str = 'tanh', 
                 return_sequences: bool = False, name: str = None):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.activation = activation
        self.return_sequences = return_sequences
        self.rnn_cell = None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer based on input shape"""
        # input_shape: (batch_size, sequence_length, input_size)
        input_size = input_shape[-1]
        self.rnn_cell = SimpleRNNCell(input_size, self.hidden_size, self.activation)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through RNN layer
        
        Args:
            inputs: Input sequences, shape (batch_size, sequence_length, input_size)
            
        Returns:
            If return_sequences=True: (batch_size, sequence_length, hidden_size)
            If return_sequences=False: (batch_size, hidden_size)
        """
        if self.rnn_cell is None:
            self.build(inputs.shape)
        
        batch_size, seq_length, input_size = inputs.shape
        
        # Initialize hidden state
        hidden_state = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_length, self.hidden_size), dtype=np.float32)
            
            for t in range(seq_length):
                hidden_state = self.rnn_cell.forward_step(inputs[:, t, :], hidden_state)
                outputs[:, t, :] = hidden_state
            
            return outputs
        else:
            # Only return the last hidden state
            for t in range(seq_length):
                hidden_state = self.rnn_cell.forward_step(inputs[:, t, :], hidden_state)
            
            return hidden_state
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.rnn_cell is None:
            return {}
        return self.rnn_cell.get_weights()
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        if self.rnn_cell is not None:
            self.rnn_cell.set_weights(weights)