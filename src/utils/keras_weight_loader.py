from models.rnn import RNNModel

class KerasWeightLoader:
    """Utility class for loading weights from Keras models"""
    
    @staticmethod
    def load_keras_weights(model: RNNModel, keras_model) -> None:
        """
        Load weights from a Keras model into our custom RNN model
        
        Args:
            model: Our custom RNN model
            keras_model: Trained Keras model
        """
        keras_weights = keras_model.get_weights()
        keras_layer_names = [layer.name for layer in keras_model.layers]
        
        weight_mapping = {}
        weight_idx = 0
        
        for layer_name in keras_layer_names:
            keras_layer = keras_model.get_layer(layer_name)
            layer_weights = keras_layer.get_weights()
            
            if len(layer_weights) == 0:
                continue
                
            # Map based on layer type
            if 'embedding' in layer_name.lower():
                weight_mapping['embedding'] = {
                    'embedding_matrix': layer_weights[0]
                }
            elif 'simple_rnn' in layer_name.lower() or 'rnn' in layer_name.lower():
                # Simple RNN weights: [W_ih, W_hh, b_h]
                if len(layer_weights) >= 3:
                    weight_mapping[layer_name] = {
                        'W_ih': layer_weights[0].T,  # Transpose for our convention
                        'W_hh': layer_weights[1].T,
                        'b_h': layer_weights[2]
                    }
            elif 'bidirectional' in layer_name.lower():
                # Bidirectional layer has weights for both directions
                if len(layer_weights) >= 6:
                    weight_mapping[layer_name] = {
                        'forward_W_ih': layer_weights[0].T,
                        'forward_W_hh': layer_weights[1].T,
                        'forward_b_h': layer_weights[2],
                        'backward_W_ih': layer_weights[3].T,
                        'backward_W_hh': layer_weights[4].T,
                        'backward_b_h': layer_weights[5]
                    }
            elif 'dense' in layer_name.lower():
                # Dense layer weights: [W, b]
                if len(layer_weights) >= 2:
                    weight_mapping[layer_name] = {
                        'W': layer_weights[0].T,  # Transpose for our convention
                        'b': layer_weights[1]
                    }
        
        model.set_weights(weight_mapping)