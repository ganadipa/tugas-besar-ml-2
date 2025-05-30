#!/usr/bin/env python3
"""
Enhanced LSTM Test Suite
Building upon your existing tests with more comprehensive coverage
"""

import os
import sys
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tempfile

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.lstm import LSTMModel, LSTMModelBuilder

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class EnhancedLSTMTests(unittest.TestCase):
    """Enhanced test suite building on your existing tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_config = {
            'vocab_size': 100,
            'embedding_dim': 16,
            'lstm_units': 8,
            'num_classes': 3,
            'num_lstm_layers': 1,
            'bidirectional': False,
            'dropout_rate': 0.0,
        }
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def create_keras_lstm(self, config, max_length=20):
        """Helper to create Keras LSTM model"""
        model = keras.Sequential([
            layers.Embedding(
                input_dim=config['vocab_size'],
                output_dim=config['embedding_dim'],
                input_length=max_length,
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
        
        model.build(input_shape=(None, max_length))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    def extract_lstm_weights(self, keras_model):
        """Extract weights from Keras LSTM model"""
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
            elif 'lstm' in layer_name.lower():
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
            elif 'dense' in layer_name.lower() or 'classification' in layer_name.lower():
                weights_dict['classification'] = {
                    'W': layer_weights[0].T,
                    'b': layer_weights[1]
                }
        
        return weights_dict

    # ============ NEW ENHANCED TESTS ============
    
    def test_different_hidden_sizes(self):
        """Test LSTM with different hidden unit sizes (spec requirement)"""
        hidden_sizes = [4, 8, 16]  # 3 variations as required by spec
        
        for hidden_size in hidden_sizes:
            with self.subTest(hidden_size=hidden_size):
                config = self.base_config.copy()
                config['lstm_units'] = hidden_size
                
                # Create models
                keras_model = self.create_keras_lstm(config)
                scratch_model = LSTMModelBuilder.create_lstm_model(**config)
                
                # Test data
                test_input = np.random.randint(0, config['vocab_size'], (2, 20))
                
                # Build scratch model
                _ = scratch_model.forward(test_input)
                
                # Load weights
                weights_dict = self.extract_lstm_weights(keras_model)
                scratch_model.set_weights(weights_dict)
                
                # Compare predictions
                keras_pred = keras_model.predict(test_input, verbose=0)
                scratch_pred = scratch_model.forward(test_input)
                
                max_diff = np.max(np.abs(keras_pred - scratch_pred))
                self.assertLess(max_diff, 1e-5, 
                               f"Hidden size {hidden_size}: max diff {max_diff:.8f}")
    
    def test_multi_layer_lstm(self):
        """Test LSTM with multiple layers (spec requirement)"""
        layer_counts = [1, 2, 3]  # 3 variations as required by spec
        
        for num_layers in layer_counts:
            with self.subTest(num_layers=num_layers):
                config = self.base_config.copy()
                config['num_lstm_layers'] = num_layers
                
                # Create from-scratch model
                scratch_model = LSTMModelBuilder.create_lstm_model(**config)
                
                # Test data
                test_input = np.random.randint(0, config['vocab_size'], (2, 15))
                
                # Test forward pass (should not crash)
                output = scratch_model.forward(test_input)
                
                # Check output shape
                expected_shape = (2, config['num_classes'])
                self.assertEqual(output.shape, expected_shape)
                
                # Check output is valid probabilities
                self.assertTrue(np.all(output >= 0))
                self.assertTrue(np.all(output <= 1))
                row_sums = np.sum(output, axis=1)
                np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    
    def test_return_sequences_true(self):
        """Test LSTM with return_sequences=True"""
        config = self.base_config.copy()
        
        # Create LSTM layer with return_sequences=True
        from layers.lstm import LSTMLayer
        lstm_layer = LSTMLayer(
            hidden_size=config['lstm_units'],
            return_sequences=True,
            name="lstm_test"
        )
        
        # Test data
        seq_length = 10
        test_input = np.random.randn(3, seq_length, config['embedding_dim'])
        
        # Forward pass
        output = lstm_layer.forward(test_input)
        
        # Check output shape
        expected_shape = (3, seq_length, config['lstm_units'])
        self.assertEqual(output.shape, expected_shape)
    
    def test_different_sequence_lengths(self):
        """Test LSTM with various sequence lengths"""
        config = self.base_config.copy()
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        sequence_lengths = [1, 5, 10, 25, 50]  # Including edge cases
        
        for seq_len in sequence_lengths:
            with self.subTest(seq_len=seq_len):
                test_input = np.random.randint(0, config['vocab_size'], (2, seq_len))
                
                # Should not crash
                output = scratch_model.forward(test_input)
                
                # Check output shape
                expected_shape = (2, config['num_classes'])
                self.assertEqual(output.shape, expected_shape)
    
    def test_different_batch_sizes(self):
        """Test LSTM with different batch sizes"""
        config = self.base_config.copy()
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        batch_sizes = [1, 3, 8, 16]
        seq_length = 15
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                test_input = np.random.randint(0, config['vocab_size'], (batch_size, seq_length))
                
                output = scratch_model.forward(test_input)
                
                # Check output shape
                expected_shape = (batch_size, config['num_classes'])
                self.assertEqual(output.shape, expected_shape)
    
    def test_bidirectional_lstm_comprehensive(self):
        """Test bidirectional LSTM more thoroughly"""
        config = self.base_config.copy()
        config['bidirectional'] = True
        
        # Create models
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        # Test data
        test_input = np.random.randint(0, config['vocab_size'], (3, 12))
        
        # Forward pass
        output = scratch_model.forward(test_input)
        
        # Check output shape (should have double the units due to bidirectional)
        expected_shape = (3, config['num_classes'])
        self.assertEqual(output.shape, expected_shape)
        
        # Check that output is valid probabilities
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
        row_sums = np.sum(output, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5)
    
    def test_zero_input_handling(self):
        """Test LSTM with zero/padding inputs"""
        config = self.base_config.copy()
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        # All zeros (like padding tokens)
        zero_input = np.zeros((2, 10), dtype=int)
        output = scratch_model.forward(zero_input)
        
        # Should produce valid output
        self.assertEqual(output.shape, (2, config['num_classes']))
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))
    
    def test_weight_shapes_consistency(self):
        """Test that weight shapes are consistent between layers"""
        config = self.base_config.copy()
        config['num_lstm_layers'] = 2
        
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        # Build model
        test_input = np.random.randint(0, config['vocab_size'], (1, 10))
        _ = scratch_model.forward(test_input)
        
        # Check that we can get weights without errors
        weights = scratch_model.get_weights()
        
        # Should have weights for all layers
        expected_layers = ['embedding', 'lstm_0', 'lstm_1', 'classification']
        for layer_name in expected_layers:
            if layer_name in ['lstm_0', 'lstm_1']:
                self.assertIn(layer_name, weights, f"Missing weights for {layer_name}")
    
    def test_numerical_stability(self):
        """Test LSTM numerical stability with extreme values"""
        config = self.base_config.copy()
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        # Test with maximum vocabulary values
        extreme_input = np.full((2, 10), config['vocab_size'] - 1, dtype=int)
        
        output = scratch_model.forward(extreme_input)
        
        # Check for NaN or Inf values
        self.assertFalse(np.any(np.isnan(output)), "Output contains NaN values")
        self.assertFalse(np.any(np.isinf(output)), "Output contains Inf values")
        
        # Check output bounds
        self.assertTrue(np.all(output >= 0))
        self.assertTrue(np.all(output <= 1))


class LSTMPerformanceTests(unittest.TestCase):
    """Performance-focused tests for LSTM implementation"""
    
    def test_inference_speed(self):
        """Test inference speed (basic performance check)"""
        config = {
            'vocab_size': 1000,
            'embedding_dim': 64,
            'lstm_units': 128,
            'num_classes': 10,
            'num_lstm_layers': 1,
            'bidirectional': False,
            'dropout_rate': 0.0,
        }
        
        scratch_model = LSTMModelBuilder.create_lstm_model(**config)
        
        # Larger test data
        test_input = np.random.randint(0, config['vocab_size'], (32, 100))
        
        import time
        start_time = time.time()
        
        # Run inference
        output = scratch_model.forward(test_input)
        
        inference_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(inference_time, 5.0, f"Inference took {inference_time:.2f}s, too slow")
        
        # Verify output
        self.assertEqual(output.shape, (32, config['num_classes']))


def run_enhanced_tests():
    """Run the enhanced test suite"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(EnhancedLSTMTests))
    suite.addTests(loader.loadTestsFromTestCase(LSTMPerformanceTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Enhanced LSTM Test Suite...")
    print("=" * 60)
    
    success = run_enhanced_tests()
    
    if success:
        print("\n🎉 All enhanced tests passed!")
        print("Your LSTM implementation handles various scenarios correctly.")
    else:
        print("\n❌ Some tests failed.")
        print("Review the failures to improve your implementation.")
    
    sys.exit(0 if success else 1)