#!/usr/bin/env python3
"""
FIXED RNN Training and From-Scratch Implementation Script
IF3270 Machine Learning Assignment 2

This script implements:
1. RNN model training with Keras using TextVectorization
2. Weight saving and loading
3. From-scratch RNN implementation with same architecture
4. Forward propagation comparison between Keras and from-scratch models
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import json
import time
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append('src')

from utils.data_loader import DataLoader
from utils.metrics_calculator import MetricsCalculator
from models.rnn import RNNModel, RNNModelBuilder
from playgrounds.starter import Starter

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class KerasRNNExperiment:
    """Keras RNN experiment class for systematic hyperparameter analysis"""
    
    def __init__(self, data_loader, X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.data_loader = data_loader
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        
        # Base configuration updated for TextVectorization compatibility
        self.base_config = {
            'vocab_size': data_loader.preprocessor.vocab_size,
            'embedding_dim': 64,
            'rnn_units': 32,
            'num_classes': data_loader.num_classes,
            'max_length': data_loader.preprocessor.max_length,
            'activation': 'tanh',
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 15
        }
        
        print(f"🏗️ Base configuration:")
        for key, value in self.base_config.items():
            print(f"  {key}: {value}")
    
    def create_keras_model(self, config):
        """Create Keras RNN model with given configuration"""
        model = keras.Sequential()
        
        # Embedding layer - input should be integer sequences from TextVectorization
        model.add(layers.Embedding(
            input_dim=config['vocab_size'],
            output_dim=config['embedding_dim'],
            input_length=config['max_length'],
            name='embedding'
        ))
        
        # RNN layers
        for i in range(config['num_rnn_layers']):
            return_sequences = i < config['num_rnn_layers'] - 1
            
            rnn_layer = layers.SimpleRNN(
                config['rnn_units'],
                activation=config['activation'],
                return_sequences=return_sequences,
                name=f'simple_rnn_{i}'
            )
            
            if config['bidirectional']:
                model.add(layers.Bidirectional(rnn_layer, name=f'bidirectional_rnn_{i}'))
            else:
                model.add(rnn_layer)
            
            # Add dropout after each RNN layer except the last
            if i < config['num_rnn_layers'] - 1:
                model.add(layers.Dropout(config['dropout_rate'], name=f'dropout_{i}'))
        
        # Final dropout and classification layer
        model.add(layers.Dropout(config['dropout_rate'], name='dropout_final'))
        model.add(layers.Dense(config['num_classes'], activation='softmax', name='classification'))
        
        # Build model with proper input shape
        model.build(input_shape=(None, config['max_length']))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def safe_f1_score(self, y_true, y_pred, average='macro'):
        """Calculate F1 score with safety checks"""
        try:
            unique_pred = np.unique(y_pred)
            unique_true = np.unique(y_true)
            
            if len(unique_pred) == 1 or len(unique_true) == 1:
                print(f"Warning: Limited class diversity in predictions or true labels")
                accuracy = np.mean(y_true == y_pred)
                print(f"Using accuracy as fallback: {accuracy:.4f}")
                return accuracy
            
            return f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            return np.mean(y_true == y_pred)
    
    def train_and_evaluate(self, config, experiment_name):
        """Train and evaluate a single configuration"""
        print(f"\n{'='*60}")
        print(f"🚀 Training: {experiment_name}")
        print(f"⚙️ Config: {config}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = self.create_keras_model(config)
        print(f"📊 Model created with {model.count_params():,} parameters")
        
        # Train model with validation monitoring
        print(f"🏃 Starting training for {config['epochs']} epochs...")
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluate on test set
        print(f"📝 Evaluating on test set...")
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        test_predictions = model.predict(self.X_test, verbose=0)
        test_pred_classes = np.argmax(test_predictions, axis=1)
        
        # Calculate F1 scores
        test_f1_macro = self.safe_f1_score(self.y_test, test_pred_classes, average='macro')
        
        # Evaluate on validation set
        valid_predictions = model.predict(self.X_valid, verbose=0)
        valid_pred_classes = np.argmax(valid_predictions, axis=1)
        valid_f1_macro = self.safe_f1_score(self.y_valid, valid_pred_classes, average='macro')
        
        # Save weights for from-scratch comparison
        weights_path = f"results/{experiment_name}_weights.npz"
        os.makedirs("results", exist_ok=True)
        self.save_keras_weights(model, weights_path, config)
        
        training_time = time.time() - start_time
        
        print(f"\n📊 Results for {experiment_name}:")
        print(f"  ⏱️ Training time: {training_time:.2f} seconds")
        print(f"  🎯 Test Accuracy: {test_acc:.4f}")
        print(f"  🎯 Test F1-Score (macro): {test_f1_macro:.4f}")
        print(f"  🎯 Valid F1-Score (macro): {valid_f1_macro:.4f}")
        print(f"  💾 Weights saved to: {weights_path}")
        
        return {
            'model': model,
            'history': history.history,
            'test_accuracy': test_acc,
            'test_f1_score': test_f1_macro,
            'valid_f1_score': valid_f1_macro,
            'weights_path': weights_path,
            'config': config,
            'training_time': training_time
        }
    
    def save_keras_weights(self, model, filepath, config):
        """FIXED: Save Keras weights in format compatible with LSTM from-scratch model"""
        print(f"💾 Saving Keras weights to: {filepath}")
        
        try:
            if len(model.weights) == 0:
                raise ValueError("Model has no weights to save!")
            
            weights_dict = {}
            lstm_layer_count = 0
            
            for layer in model.layers:
                layer_weights = layer.get_weights()
                if len(layer_weights) == 0:
                    continue
                    
                layer_name = layer.name
                print(f"  Processing layer: {layer_name} - {len(layer_weights)} weight arrays")
                
                if 'embedding' in layer_name.lower():
                    weights_dict['embedding'] = {
                        'embedding_matrix': layer_weights[0]
                    }
                elif 'lstm' in layer_name.lower() and 'bidirectional' not in layer_name.lower():
                    # Unidirectional LSTM
                    if len(layer_weights) >= 3:
                        kernel = layer_weights[0]          # Input weights
                        recurrent_kernel = layer_weights[1] # Recurrent weights  
                        bias = layer_weights[2]             # Bias
                        
                        hidden_size = kernel.shape[1] // 4
                        print(f"    LSTM hidden size: {hidden_size}")
                        
                        # CRITICAL: Keras order is i, f, c, o (input, forget, cell, output)
                        W_i = kernel[:, :hidden_size].T                    # Input gate
                        W_f = kernel[:, hidden_size:2*hidden_size].T       # Forget gate  
                        W_c = kernel[:, 2*hidden_size:3*hidden_size].T     # Cell gate
                        W_o = kernel[:, 3*hidden_size:].T                  # Output gate
                        
                        U_i = recurrent_kernel[:, :hidden_size].T          # Input gate recurrent
                        U_f = recurrent_kernel[:, hidden_size:2*hidden_size].T    # Forget gate recurrent
                        U_c = recurrent_kernel[:, 2*hidden_size:3*hidden_size].T  # Cell gate recurrent  
                        U_o = recurrent_kernel[:, 3*hidden_size:].T        # Output gate recurrent
                        
                        b_i = bias[:hidden_size]                           # Input gate bias
                        b_f = bias[hidden_size:2*hidden_size]             # Forget gate bias
                        b_c = bias[2*hidden_size:3*hidden_size]           # Cell gate bias
                        b_o = bias[3*hidden_size:]                        # Output gate bias
                        
                        target_name = f'lstm_{lstm_layer_count}'
                        weights_dict[target_name] = {
                            # Input gate
                            'W_ii': W_i, 'W_hi': U_i, 'b_i': b_i,
                            # Forget gate  
                            'W_if': W_f, 'W_hf': U_f, 'b_f': b_f,
                            # Cell gate (candidate values)
                            'W_ig': W_c, 'W_hg': U_c, 'b_g': b_c,
                            # Output gate
                            'W_io': W_o, 'W_ho': U_o, 'b_o': b_o,
                        }
                        lstm_layer_count += 1
                elif 'bidirectional' in layer_name.lower():
                    # Bidirectional LSTM
                    if len(layer_weights) >= 6:
                        # Forward direction
                        forward_kernel = layer_weights[0]
                        forward_recurrent = layer_weights[1] 
                        forward_bias = layer_weights[2]
                        
                        # Backward direction
                        backward_kernel = layer_weights[3]
                        backward_recurrent = layer_weights[4]
                        backward_bias = layer_weights[5]
                        
                        hidden_size = forward_kernel.shape[1] // 4
                        print(f"    Bidirectional LSTM hidden size: {hidden_size}")
                        
                        # Extract forward weights (i, f, c, o order)
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
                        
                        # Extract backward weights (i, f, c, o order)
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
                        
                        target_name = f'bidirectional_lstm_{lstm_layer_count}'
                        weights_dict[target_name] = {
                            # Forward weights
                            'forward_W_ii': fW_i, 'forward_W_hi': fU_i, 'forward_b_i': fb_i,
                            'forward_W_if': fW_f, 'forward_W_hf': fU_f, 'forward_b_f': fb_f,
                            'forward_W_ig': fW_c, 'forward_W_hg': fU_c, 'forward_b_g': fb_c,
                            'forward_W_io': fW_o, 'forward_W_ho': fU_o, 'forward_b_o': fb_o,
                            # Backward weights
                            'backward_W_ii': bW_i, 'backward_W_hi': bU_i, 'backward_b_i': bb_i,
                            'backward_W_if': bW_f, 'backward_W_hf': bU_f, 'backward_b_f': bb_f,
                            'backward_W_ig': bW_c, 'backward_W_hg': bU_c, 'backward_b_g': bb_c,
                            'backward_W_io': bW_o, 'backward_W_ho': bU_o, 'backward_b_o': bb_o,
                        }
                        lstm_layer_count += 1
                elif 'dense' in layer_name.lower() or 'classification' in layer_name.lower():
                    weights_dict['classification'] = {
                        'W': layer_weights[0].T,
                        'b': layer_weights[1]
                    }
            
            # Save weights
            save_dict = {}
            for layer_name, layer_weights in weights_dict.items():
                for weight_name, weight_value in layer_weights.items():
                    save_dict[f"{layer_name}_{weight_name}"] = weight_value
            
            # Also save configuration for reference
            save_dict['config'] = json.dumps(config)
            
            np.savez(filepath, **save_dict)
            print(f"  ✅ Saved {len(save_dict)-1} weight arrays successfully")
            print(f"  📝 Layers saved: {list(weights_dict.keys())}")
            
            # Verify what was saved
            loaded_check = np.load(filepath)
            print(f"  🔍 Verification - Keys in saved file: {list(loaded_check.keys())[:10]}...")
            loaded_check.close()
            
        except Exception as e:
            print(f"❌ Error saving weights: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution function"""
    print("RNN Training and From-Scratch Implementation")
    print("IF3270 Machine Learning Assignment 2")
    print("=" * 60)
    
    # Initialize starter
    starter = Starter()
    
    def run_experiment():
        try:
            print("1. LOADING AND PREPROCESSING DATA")
            print("="*50)
            
            # Initialize data loader
            data_dir = "../data"
            data_loader = DataLoader(data_dir)
            
            # Load and prepare data using Keras TextVectorization
            X_train, y_train, X_valid, y_valid, X_test, y_test = data_loader.prepare_data(
                max_vocab_size=5000,
                max_length=50
            )
            
            print(f"\n✅ Data loaded successfully!")
            print(f"📊 Vocabulary size: {data_loader.preprocessor.vocab_size}")
            print(f"📊 Number of classes: {data_loader.num_classes}")
            print(f"📊 Max sequence length: {data_loader.preprocessor.max_length}")
            
            # Show class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"\n📈 Class distribution in training data:")
            for class_id, count in zip(unique, counts):
                class_name = data_loader.reverse_label_encoder[class_id]
                print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")
            
            # Initialize experiment runner
            print("\n🏗️ Initializing Keras experiment framework...")
            keras_experiment = KerasRNNExperiment(
                data_loader, X_train, y_train, X_valid, y_valid, X_test, y_test
            )
            
            print("\n2. HYPERPARAMETER ANALYSIS")
            print("="*50)
            
            # Layer count analysis
            print("\n2.1 Analyzing Effect of RNN Layer Count")
            print("-"*40)
            layer_counts = [1, 2, 3]
            layer_results = {}
            
            for num_layers in layer_counts:
                config = keras_experiment.base_config.copy()
                config.update({
                    'num_rnn_layers': num_layers,
                    'bidirectional': False,
                    'rnn_units': 64
                })
                
                experiment_name = f"rnn_layers_{num_layers}"
                try:
                    result = keras_experiment.train_and_evaluate(config, experiment_name)
                    layer_results[num_layers] = result
                    print(f"✅ {experiment_name} completed successfully!")
                except Exception as e:
                    print(f"❌ Error in {experiment_name}: {e}")
                    continue
            
            # Units analysis
            print("\n2.2 Analyzing Effect of RNN Units per Layer")
            print("-"*40)
            unit_counts = [32, 64, 128]
            unit_results = {}
            
            for num_units in unit_counts:
                config = keras_experiment.base_config.copy()
                config.update({
                    'num_rnn_layers': 1,
                    'rnn_units': num_units,
                    'bidirectional': False
                })
                
                experiment_name = f"rnn_units_{num_units}"
                try:
                    result = keras_experiment.train_and_evaluate(config, experiment_name)
                    unit_results[num_units] = result
                    print(f"✅ {experiment_name} completed successfully!")
                except Exception as e:
                    print(f"❌ Error in {experiment_name}: {e}")
                    continue
            
            # Direction analysis
            print("\n2.3 Analyzing Effect of Bidirectional RNN")
            print("-"*40)
            direction_results = {}
            
            for is_bidirectional in [False, True]:
                config = keras_experiment.base_config.copy()
                config.update({
                    'num_rnn_layers': 1,
                    'rnn_units': 64,
                    'bidirectional': is_bidirectional
                })
                
                direction_name = "bidirectional" if is_bidirectional else "unidirectional"
                experiment_name = f"rnn_{direction_name}"
                
                try:
                    result = keras_experiment.train_and_evaluate(config, experiment_name)
                    direction_results[direction_name] = result
                    print(f"✅ {experiment_name} completed successfully!")
                except Exception as e:
                    print(f"❌ Error in {experiment_name}: {e}")
                    continue
            
            # From-scratch comparison demonstration
            print("\n4. FROM-SCRATCH MODEL IMPLEMENTATION AND COMPARISON")
            print("="*60)
            
            if layer_results:
                # Use best layer configuration for comparison
                best_layers = max(layer_results.keys(), key=lambda x: layer_results[x]['test_f1_score'])
                best_result = layer_results[best_layers]
                
                print(f"🔍 Using best layer configuration: {best_layers} layers")
                print(f"📊 Best F1-score: {best_result['test_f1_score']:.4f}")
                
                # Create from-scratch model
                config = best_result['config']
                scratch_model = RNNModelBuilder.create_simple_rnn_model(
                    vocab_size=config['vocab_size'],
                    embedding_dim=config['embedding_dim'],
                    rnn_units=config['rnn_units'],
                    num_classes=config['num_classes'],
                    num_rnn_layers=config['num_rnn_layers'],
                    bidirectional=config['bidirectional'],
                    dropout_rate=config['dropout_rate'],
                    activation=config['activation']
                )
                
                # Build the scratch model
                dummy_input = X_test[:1]
                _ = scratch_model.forward(dummy_input)
                
                print("\nFrom-scratch model summary:")
                print(scratch_model.summary())
                
                # Load weights
                try:
                    weights_path = best_result['weights_path']
                    scratch_model.load_weights(weights_path)
                    print(f"✅ Weights loaded successfully from: {weights_path}")
                    
                    # Compare predictions
                    test_sample = X_test[:100]  # Use subset for comparison
                    keras_pred = best_result['model'].predict(test_sample, verbose=0)
                    scratch_pred = scratch_model.forward(test_sample)
                    
                    max_diff = np.max(np.abs(keras_pred - scratch_pred))
                    mean_diff = np.mean(np.abs(keras_pred - scratch_pred))
                    
                    print(f"\n📊 COMPARISON RESULTS:")
                    print(f"  Maximum difference: {max_diff:.8f}")
                    print(f"  Mean difference: {mean_diff:.8f}")
                    
                    if max_diff < 1e-5:
                        print("✅ EXCELLENT: Models produce nearly identical outputs!")
                    elif max_diff < 1e-3:
                        print("✅ GOOD: Models produce similar outputs with small differences.")
                    else:
                        print("⚠️ WARNING: Models have significant differences in outputs.")
                    
                except Exception as e:
                    print(f"❌ Error in from-scratch comparison: {e}")
                    import traceback
                    traceback.print_exc()
            
            print("\n" + "=" * 60)
            print("EXPERIMENT COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error in experiment: {e}")
            import traceback
            traceback.print_exc()
    
    # Initialize starter and run experiment
    starter.start(run_experiment)

if __name__ == "__main__":
    main()