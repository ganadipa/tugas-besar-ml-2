#!/usr/bin/env python3
"""
LSTM Implementation Test Suite
Tests for LSTM model correctness against Keras implementation

This test suite ensures that our from-scratch LSTM implementation
produces identical results to Keras LSTM models.
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.lstm import LSTMModel, LSTMModelBuilder

# Suppress TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class TestLSTMImplementation(unittest.TestCase):
    """Test suite for LSTM implementation correctness"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.config = {
            'vocab_size': 100,
            'embedding_dim': 16,
            'lstm_units': 8,
            'num_classes': 3,
            'max_length': 20,
            'num_lstm_layers': 1,
            'bidirectional': False,
            'dropout_rate': 0.0,
        }
        
        # Create test data
        self.batch_size = 3
        self.test_input = np.random.randint(
            0, self.config['vocab_size'], 
            (self.batch_size, self.config['max_length'])
        )
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def create_keras_model(self, config):
        """Create a Keras LSTM model with given configuration"""
        model = keras.Sequential([
            layers.Embedding(
                input_dim=config['vocab_size'],
                output_dim=config['embedding_dim'],
                input_length=config['max_length'],
                name='embedding'
            ),
            layers.LSTM(
                config['lstm_units'],
                return_sequences=False,
                name='lstm_0'
            ),
            layers.Dense(
                config['num_classes'],
                activation='softmax',
                name='classification'
            )
        ])
        
        model.build(input_shape=(None, config['max_length']))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    def create_bidirectional_keras_model(self, config):
        """Create a bidirectional Keras LSTM model"""
        model = keras.Sequential([
            layers.Embedding(
                input_dim=config['vocab_size'],
                output_dim=config['embedding_dim'],
                input_length=config['max_length'],
                name='embedding'
            ),
            layers.Bidirectional(
                layers.LSTM(config['lstm_units'], return_sequences=False),
                name='bidirectional_lstm_0'
            ),
            layers.Dense(
                config['num_classes'],
                activation='softmax',
                name='classification'
            )
        ])
        
        model.build(input_shape=(None, config['max_length']))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    def extract_lstm_weights(self, keras_model):
        """Extract weights from Keras LSTM model with correct gate order"""
        weights_dict = {}
        
        for layer in keras_model.layers:
            layer_weights = layer.get_weights()
            if len(layer_weights) == 0:
                continue
                
            layer_name = layer.name
            
            if 'embedding' in layer_name.lower():
                weights_dict['embedding'] = {
                    'embedding_matrix': layer_weights[0]
                }
            elif 'lstm' in layer_name.lower() and 'bidirectional' not in layer_name.lower():
                if len(layer_weights) >= 3:
                    kernel = layer_weights[0]
                    recurrent_kernel = layer_weights[1]
                    bias = layer_weights[2]
                    
                    hidden_size = kernel.shape[1] // 4
                    
                    # Keras order: input, forget, cell, output (i, f, c, o)
                    W_i = kernel[:, :hidden_size].T
                    W_f = kernel[:, hidden_size:2*hidden_size].T
                    W_c = kernel[:, 2*hidden_size:3*hidden_size].T
                    W_o = kernel[:, 3*hidden_size:].T
                    
                    U_i = recurrent_kernel[:, :hidden_size].T
                    U_f = recurrent_kernel[:, hidden_size:2*hidden_size].T
                    U_c = recurrent_kernel[:, 2*hidden_size:3*hidden_size].T
                    U_o = recurrent_kernel[:, 3*hidden_size:].T
                    
                    b_i = bias[:hidden_size]
                    b_f = bias[hidden_size:2*hidden_size]
                    b_c = bias[2*hidden_size:3*hidden_size]
                    b_o = bias[3*hidden_size:]
                    
                    weights_dict['lstm_0'] = {
                        'W_if': W_f, 'W_hf': U_f, 'b_f': b_f,
                        'W_ii': W_i, 'W_hi': U_i, 'b_i': b_i,
                        'W_ig': W_c, 'W_hg': U_c, 'b_g': b_c,
                        'W_io': W_o, 'W_ho': U_o, 'b_o': b_o,
                    }
            elif 'bidirectional' in layer_name.lower():
                if len(layer_weights) >= 6:
                    forward_kernel = layer_weights[0]
                    forward_recurrent = layer_weights[1]
                    forward_bias = layer_weights[2]
                    backward_kernel = layer_weights[3]
                    backward_recurrent = layer_weights[4]
                    backward_bias = layer_weights[5]
                    
                    hidden_size = forward_kernel.shape[1] // 4
                    
                    # Forward direction
                    fW_i = forward_kernel[:, :hidden_size].T
                    fW_f = forward_kernel[:, hidden_size:2*hidden_size].T
                    fW_c = forward_kernel[:, 2*hidden_size:3*hidden_size].T
                    fW_o = forward_kernel[:, 3*hidden_size:].T
                    
                    fU_i = forward_recurrent[:, :hidden_size].T
                    fU_f = forward_recurrent[:, hidden_size:2*hidden_size].T
                    fU_c = forward_recurrent[:, 2*hidden_size:3*hidden_size].T
                    fU_o = forward_recurrent[:, 3*hidden_size:].T
                    
                    fb_i = forward_bias[:hidden_size]
                    fb_f = forward_bias[hidden_size:2*hidden_size]
                    fb_c = forward_bias[2*hidden_size:3*hidden_size]
                    fb_o = forward_bias[3*hidden_size:]
                    
                    # Backward direction
                    bW_i = backward_kernel[:, :hidden_size].T
                    bW_f = backward_kernel[:, hidden_size:2*hidden_size].T
                    bW_c = backward_kernel[:, 2*hidden_size:3*hidden_size].T
                    bW_o = backward_kernel[:, 3*hidden_size:].T
                    
                    bU_i = backward_recurrent[:, :hidden_size].T
                    bU_f = backward_recurrent[:, hidden_size:2*hidden_size].T
                    bU_c = backward_recurrent[:, 2*hidden_size:3*hidden_size].T
                    bU_o = backward_recurrent[:, 3*hidden_size:].T
                    
                    bb_i = backward_bias[:hidden_size]
                    bb_f = backward_bias[hidden_size:2*hidden_size]
                    bb_c = backward_bias[2*hidden_size:3*hidden_size]
                    bb_o = backward_bias[3*hidden_size:]
                    
                    weights_dict['bidirectional_lstm_0'] = {
                        # Forward weights
                        'forward_W_if': fW_f, 'forward_W_hf': fU_f, 'forward_b_f': fb_f,
                        'forward_W_ii': fW_i, 'forward_W_hi': fU_i, 'forward_b_i': fb_i,
                        'forward_W_ig': fW_c, 'forward_W_hg': fU_c, 'forward_b_g': fb_c,
                        'forward_W_io': fW_o, 'forward_W_ho': fU_o, 'forward_b_o': fb_o,
                        # Backward weights
                        'backward_W_if': bW_f, 'backward_W_hf': bU_f, 'backward_b_f': bb_f,
                        'backward_W_ii': bW_i, 'backward_W_hi': bU_i, 'backward_b_i': bb_i,
                        'backward_W_ig': bW_c, 'backward_W_hg': bU_c, 'backward_b_g': bb_c,
                        'backward_W_io': bW_o, 'backward_W_ho': bU_o, 'backward_b_o': bb_o,
                    }
            elif 'dense' in layer_name.lower() or 'classification' in layer_name.lower():
                weights_dict['classification'] = {
                    'W': layer_weights[0].T,
                    'b': layer_weights[1]
                }
        
        return weights_dict
    
    def test_unidirectional_lstm_direct_loading(self):
        """Test direct weight loading for unidirectional LSTM"""
        # Create models
        keras_model = self.create_keras_model(self.config)
        scratch_model = LSTMModelBuilder.create_lstm_model(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            lstm_units=self.config['lstm_units'],
            num_classes=self.config['num_classes'],
            num_lstm_layers=self.config['num_lstm_layers'],
            bidirectional=self.config['bidirectional'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Build scratch model
        _ = scratch_model.forward(self.test_input)
        
        # Extract and load weights
        weights_dict = self.extract_lstm_weights(keras_model)
        scratch_model.set_weights(weights_dict)
        
        # Test predictions
        keras_pred = keras_model.predict(self.test_input, verbose=0)
        scratch_pred = scratch_model.forward(self.test_input)
        
        max_diff = np.max(np.abs(keras_pred - scratch_pred))
        
        self.assertLess(max_diff, 1e-5, 
                       f"LSTM predictions differ by {max_diff:.8f}, expected < 1e-5")
    
    def test_unidirectional_lstm_file_loading(self):
        """Test file-based weight loading for unidirectional LSTM"""
        # Create models
        keras_model = self.create_keras_model(self.config)
        scratch_model = LSTMModelBuilder.create_lstm_model(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            lstm_units=self.config['lstm_units'],
            num_classes=self.config['num_classes'],
            num_lstm_layers=self.config['num_lstm_layers'],
            bidirectional=self.config['bidirectional'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Build scratch model
        _ = scratch_model.forward(self.test_input)
        
        # Create temporary file for weights
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            weights_path = tmp_file.name
        
        try:
            # Extract and save weights
            weights_dict = self.extract_lstm_weights(keras_model)
            save_dict = {}
            for layer_name, layer_weights in weights_dict.items():
                for weight_name, weight_value in layer_weights.items():
                    save_dict[f"{layer_name}_{weight_name}"] = weight_value
            
            np.savez(weights_path, **save_dict)
            
            # Load weights from file
            scratch_model.load_weights(weights_path)
            
            # Test predictions
            keras_pred = keras_model.predict(self.test_input, verbose=0)
            scratch_pred = scratch_model.forward(self.test_input)
            
            max_diff = np.max(np.abs(keras_pred - scratch_pred))
            
            self.assertLess(max_diff, 1e-5, 
                           f"LSTM file loading predictions differ by {max_diff:.8f}, expected < 1e-5")
        
        finally:
            # Clean up temporary file
            if os.path.exists(weights_path):
                os.remove(weights_path)
    
    def test_bidirectional_lstm(self):
        """Test bidirectional LSTM implementation"""
        # Update config for bidirectional
        bidirectional_config = self.config.copy()
        bidirectional_config['bidirectional'] = True
        
        # Create models
        keras_model = self.create_bidirectional_keras_model(bidirectional_config)
        scratch_model = LSTMModelBuilder.create_lstm_model(
            vocab_size=bidirectional_config['vocab_size'],
            embedding_dim=bidirectional_config['embedding_dim'],
            lstm_units=bidirectional_config['lstm_units'],
            num_classes=bidirectional_config['num_classes'],
            num_lstm_layers=bidirectional_config['num_lstm_layers'],
            bidirectional=bidirectional_config['bidirectional'],
            dropout_rate=bidirectional_config['dropout_rate']
        )
        
        # Build scratch model
        _ = scratch_model.forward(self.test_input)
        
        # Extract and load weights
        weights_dict = self.extract_lstm_weights(keras_model)
        scratch_model.set_weights(weights_dict)
        
        # Test predictions
        keras_pred = keras_model.predict(self.test_input, verbose=0)
        scratch_pred = scratch_model.forward(self.test_input)
        
        max_diff = np.max(np.abs(keras_pred - scratch_pred))
        
        self.assertLess(max_diff, 1e-3, 
                       f"Bidirectional LSTM predictions differ by {max_diff:.8f}, expected < 1e-3")
    
    def test_model_shapes(self):
        """Test that model outputs have correct shapes"""
        # Create models
        keras_model = self.create_keras_model(self.config)
        scratch_model = LSTMModelBuilder.create_lstm_model(
            vocab_size=self.config['vocab_size'],
            embedding_dim=self.config['embedding_dim'],
            lstm_units=self.config['lstm_units'],
            num_classes=self.config['num_classes'],
            num_lstm_layers=self.config['num_lstm_layers'],
            bidirectional=self.config['bidirectional'],
            dropout_rate=self.config['dropout_rate']
        )
        
        # Test predictions
        keras_pred = keras_model.predict(self.test_input, verbose=0)
        scratch_pred = scratch_model.forward(self.test_input)
        
        # Check shapes
        expected_shape = (self.batch_size, self.config['num_classes'])
        self.assertEqual(keras_pred.shape, expected_shape)
        self.assertEqual(scratch_pred.shape, expected_shape)
    
    def test_weight_extraction_completeness(self):
        """Test that all necessary weights are extracted from Keras model"""
        keras_model = self.create_keras_model(self.config)
        weights_dict = self.extract_lstm_weights(keras_model)
        
        # Check that all expected layers are present
        expected_layers = ['embedding', 'lstm_0', 'classification']
        for layer in expected_layers:
            self.assertIn(layer, weights_dict, f"Missing layer: {layer}")
        
        # Check LSTM weights
        lstm_weights = weights_dict['lstm_0']
        expected_lstm_weights = [
            'W_if', 'W_hf', 'b_f',  # Forget gate
            'W_ii', 'W_hi', 'b_i',  # Input gate
            'W_ig', 'W_hg', 'b_g',  # Cell gate
            'W_io', 'W_ho', 'b_o',  # Output gate
        ]
        
        for weight in expected_lstm_weights:
            self.assertIn(weight, lstm_weights, f"Missing LSTM weight: {weight}")


class TestLSTMStability(unittest.TestCase):
    """Test LSTM implementation stability and edge cases"""
    
    def test_different_sequence_lengths(self):
        """Test LSTM with different sequence lengths"""
        config = {
            'vocab_size': 50,
            'embedding_dim': 8,
            'lstm_units': 4,
            'num_classes': 2,
            'max_length': 10,
            'num_lstm_layers': 1,
            'bidirectional': False,
            'dropout_rate': 0.0,
        }
        
        # Remove max_length from config for LSTMModelBuilder
        builder_config = {k: v for k, v in config.items() if k != 'max_length'}
        scratch_model = LSTMModelBuilder.create_lstm_model(**builder_config)
        
        # Test with different batch sizes and sequence lengths
        test_cases = [
            (1, 5),   # Single sample, short sequence
            (2, 10),  # Small batch, full length
            (5, 3),   # Larger batch, short sequence
        ]
        
        for batch_size, seq_len in test_cases:
            with self.subTest(batch_size=batch_size, seq_len=seq_len):
                test_input = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
                
                # Should not raise an exception
                output = scratch_model.forward(test_input)
                
                # Check output shape
                expected_shape = (batch_size, config['num_classes'])
                self.assertEqual(output.shape, expected_shape)
    
    def test_zero_input_handling(self):
        """Test LSTM behavior with zero inputs"""
        config = {
            'vocab_size': 50,
            'embedding_dim': 8,
            'lstm_units': 4,
            'num_classes': 2,
            'max_length': 10,
            'num_lstm_layers': 1,
            'bidirectional': False,
            'dropout_rate': 0.0,
        }
        
        # Remove max_length from config for LSTMModelBuilder  
        builder_config = {k: v for k, v in config.items() if k != 'max_length'}
        scratch_model = LSTMModelBuilder.create_lstm_model(**builder_config)
        
        # Test with all zeros (padding tokens)
        zero_input = np.zeros((2, 10), dtype=int)
        output = scratch_model.forward(zero_input)
        
        # Should produce valid probabilities
        self.assertTrue(np.all(output >= 0), "Output should be non-negative")
        self.assertTrue(np.all(output <= 1), "Output should be <= 1")
        
        # Each sample should sum to approximately 1 (softmax output)
        row_sums = np.sum(output, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)


def run_tests():
    """Run all LSTM tests and return success status"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMImplementation))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMStability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)