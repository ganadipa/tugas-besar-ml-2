import numpy as np
from typing import Dict, Tuple
from layers.base import Layer
from activation_functions.base import ActivationFunction
from activation_functions.tanh import Tanh
from activation_functions.sigmoid import Sigmoid

class LSTMCell:
    """LSTM cell implementation with forget, input, and output gates"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for forget gate
        self.W_if = np.random.normal(0, 0.1, (hidden_size, input_size)).astype(np.float32)
        self.W_hf = np.random.normal(0, 0.1, (hidden_size, hidden_size)).astype(np.float32)
        self.b_f = np.zeros((hidden_size,), dtype=np.float32)
        
        # Initialize weights for input gate
        self.W_ii = np.random.normal(0, 0.1, (hidden_size, input_size)).astype(np.float32)
        self.W_hi = np.random.normal(0, 0.1, (hidden_size, hidden_size)).astype(np.float32)
        self.b_i = np.zeros((hidden_size,), dtype=np.float32)
        
        # Initialize weights for candidate values
        self.W_ig = np.random.normal(0, 0.1, (hidden_size, input_size)).astype(np.float32)
        self.W_hg = np.random.normal(0, 0.1, (hidden_size, hidden_size)).astype(np.float32)
        self.b_g = np.zeros((hidden_size,), dtype=np.float32)
        
        # Initialize weights for output gate
        self.W_io = np.random.normal(0, 0.1, (hidden_size, input_size)).astype(np.float32)
        self.W_ho = np.random.normal(0, 0.1, (hidden_size, hidden_size)).astype(np.float32)
        self.b_o = np.zeros((hidden_size,), dtype=np.float32)
        
        # Activation functions
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
    
    def forward_step(self, input_t: np.ndarray, hidden_t_prev: np.ndarray, 
                     cell_t_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for one time step
        
        Args:
            input_t: Input at time t, shape (batch_size, input_size)
            hidden_t_prev: Previous hidden state, shape (batch_size, hidden_size)
            cell_t_prev: Previous cell state, shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        # Forget gate
        f_t = self.sigmoid.forward(
            np.dot(input_t, self.W_if.T) + 
            np.dot(hidden_t_prev, self.W_hf.T) + 
            self.b_f
        )
        
        # Input gate
        i_t = self.sigmoid.forward(
            np.dot(input_t, self.W_ii.T) + 
            np.dot(hidden_t_prev, self.W_hi.T) + 
            self.b_i
        )
        
        # Candidate values
        g_t = self.tanh.forward(
            np.dot(input_t, self.W_ig.T) + 
            np.dot(hidden_t_prev, self.W_hg.T) + 
            self.b_g
        )
        
        # Output gate
        o_t = self.sigmoid.forward(
            np.dot(input_t, self.W_io.T) + 
            np.dot(hidden_t_prev, self.W_ho.T) + 
            self.b_o
        )
        
        # Update cell state
        cell_t = f_t * cell_t_prev + i_t * g_t
        
        # Update hidden state
        hidden_t = o_t * self.tanh.forward(cell_t)
        
        return hidden_t, cell_t
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {
            # Forget gate weights
            "W_if": self.W_if,
            "W_hf": self.W_hf,
            "b_f": self.b_f,
            # Input gate weights
            "W_ii": self.W_ii,
            "W_hi": self.W_hi,
            "b_i": self.b_i,
            # Candidate weights
            "W_ig": self.W_ig,
            "W_hg": self.W_hg,
            "b_g": self.b_g,
            # Output gate weights
            "W_io": self.W_io,
            "W_ho": self.W_ho,
            "b_o": self.b_o,
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights for the LSTM cell"""
        if "W_if" in weights: self.W_if = weights["W_if"]
        if "W_hf" in weights: self.W_hf = weights["W_hf"] 
        if "b_f" in weights: self.b_f = weights["b_f"]
        
        if "W_ii" in weights: self.W_ii = weights["W_ii"]
        if "W_hi" in weights: self.W_hi = weights["W_hi"]
        if "b_i" in weights: self.b_i = weights["b_i"]
        
        if "W_ig" in weights: self.W_ig = weights["W_ig"]
        if "W_hg" in weights: self.W_hg = weights["W_hg"]
        if "b_g" in weights: self.b_g = weights["b_g"]
        
        if "W_io" in weights: self.W_io = weights["W_io"]
        if "W_ho" in weights: self.W_ho = weights["W_ho"]
        if "b_o" in weights: self.b_o = weights["b_o"]

import numpy as np
from typing import Dict, Tuple
from layers.base import Layer

class LSTMLayer(Layer):
    """LSTM layer implementation"""
    
    def __init__(self, hidden_size: int, return_sequences: bool = False, name: str = None):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.lstm_cell = None
        
        # Store weights for loading before build
        self._pending_weights = None
    
    def build(self, input_shape: Tuple[int, ...]):
        """Build the layer based on input shape"""
        # input_shape: (batch_size, sequence_length, input_size)
        input_size = input_shape[-1]
        self.lstm_cell = LSTMCell(input_size, self.hidden_size)
        
        # Load pending weights if any
        if self._pending_weights is not None:
            print(f"Loading pending weights for {self.name}")
            self.lstm_cell.set_weights(self._pending_weights)
            self._pending_weights = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through LSTM layer
        
        Args:
            inputs: Input sequences, shape (batch_size, sequence_length, input_size)
            
        Returns:
            If return_sequences=True: (batch_size, sequence_length, hidden_size)
            If return_sequences=False: (batch_size, hidden_size)
        """
        if self.lstm_cell is None:
            self.build(inputs.shape)
        
        batch_size, seq_length, input_size = inputs.shape
        
        # Initialize hidden and cell states
        hidden_state = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        cell_state = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_length, self.hidden_size), dtype=np.float32)
            
            for t in range(seq_length):
                hidden_state, cell_state = self.lstm_cell.forward_step(
                    inputs[:, t, :], hidden_state, cell_state
                )
                outputs[:, t, :] = hidden_state
            
            return outputs
        else:
            # Only return the last hidden state
            for t in range(seq_length):
                hidden_state, cell_state = self.lstm_cell.forward_step(
                    inputs[:, t, :], hidden_state, cell_state
                )
            
            return hidden_state
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        if self.lstm_cell is None:
            return self._pending_weights or {}
        return self.lstm_cell.get_weights()
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights for the LSTM layer"""
        if self.lstm_cell is not None:
            # Cell is already built, set weights directly
            print(f"Setting weights directly for {self.name}: {list(weights.keys())}")
            self.lstm_cell.set_weights(weights)
        else:
            # Cell not built yet, store weights for later
            print(f"Storing pending weights for {self.name}: {list(weights.keys())}")
            self._pending_weights = weights.copy()

class BidirectionalLSTMLayer(Layer):
    """Bidirectional LSTM wrapper"""
    
    def __init__(self, hidden_size: int, return_sequences: bool = False, name: str = None):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        
        # Create forward and backward LSTM layers
        self.forward_layer = LSTMLayer(hidden_size, return_sequences, f"{self.name}_forward")
        self.backward_layer = LSTMLayer(hidden_size, return_sequences, f"{self.name}_backward")
        
        # Store pending weights for loading before build
        self._pending_weights = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through bidirectional LSTM layer
        
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
        
        if self.return_sequences:
            # Reverse the backward output to match forward direction
            backward_output = backward_output[:, ::-1, :]
            # Concatenate along feature dimension
            return np.concatenate([forward_output, backward_output], axis=-1)
        else:
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
        """Set weights for both forward and backward LSTM layers"""
        print(f"Setting weights for bidirectional LSTM layer {self.name}: {list(weights.keys())}")
        
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


