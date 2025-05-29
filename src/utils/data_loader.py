from utils.text_preprocessor import TextPreprocessor
import numpy as np
import pandas as pd
import os
from typing import List, Tuple
import tensorflow as tf
import keras

class DataLoader:
    """Enhanced data loading and preprocessing pipeline compatible with both Keras and from-scratch models"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.preprocessor = TextPreprocessor()  # Keep for compatibility
        self.vectorizer = None  # Keras TextVectorization layer
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.num_classes = 0
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test data"""
        train_path = os.path.join(self.data_dir, 'nusax', 'train.csv')
        valid_path = os.path.join(self.data_dir, 'nusax', 'valid.csv')
        test_path = os.path.join(self.data_dir, 'nusax', 'test.csv')
        
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded data:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Valid: {len(valid_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        return train_df, valid_df, test_df
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to integers"""
        unique_labels = sorted(set(labels))
        
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {i: label for i, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
        encoded_labels = np.array([self.label_encoder[label] for label in labels])
        
        print(f"Label encoding:")
        for label, idx in self.label_encoder.items():
            print(f"  {label}: {idx}")
        
        return encoded_labels

    def prepare_data(self, max_vocab_size: int = 10000, max_length: int = 100
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training using Keras TextVectorization
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        # Load data
        train_df, valid_df, test_df = self.load_data()

        # Collect text data
        train_texts = train_df['text'].astype(str).tolist()
        valid_texts = valid_df['text'].astype(str).tolist()
        test_texts = test_df['text'].astype(str).tolist()

        # Create TextVectorization layer
        self.vectorizer = keras.layers.TextVectorization(
            max_tokens=max_vocab_size,
            output_mode='int',
            output_sequence_length=max_length,
            standardize='lower_and_strip_punctuation',
            split='whitespace'
        )

        # Adapt vectorizer to training texts
        self.vectorizer.adapt(train_texts)

        # Extract vocabulary information for compatibility with from-scratch model
        vocab = self.vectorizer.get_vocabulary()
        self.preprocessor.vocab = {word: i for i, word in enumerate(vocab)}
        self.preprocessor.reverse_vocab = {i: word for i, word in enumerate(vocab)}
        self.preprocessor.vocab_size = len(vocab)
        self.preprocessor.max_length = max_length
        
        print(f"Keras vectorizer vocabulary size: {len(vocab)}")
        print(f"Sample vocabulary: {vocab[:10]}")

        # Vectorize datasets
        X_train = self.vectorizer(tf.constant(train_texts)).numpy()
        X_valid = self.vectorizer(tf.constant(valid_texts)).numpy()
        X_test = self.vectorizer(tf.constant(test_texts)).numpy()

        # Determine label column
        if 'label' in train_df.columns:
            label_col = 'label'
        elif 'sentiment' in train_df.columns:
            label_col = 'sentiment'
        else:
            label_col = [col for col in train_df.columns if col not in ['id', 'text']][0]

        # Encode labels
        all_labels = (train_df[label_col].astype(str).tolist() + 
                    valid_df[label_col].astype(str).tolist() + 
                    test_df[label_col].astype(str).tolist())
        unique_labels = sorted(list(set(all_labels)))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)

        y_train = np.array([self.label_encoder[str(label)] for label in train_df[label_col]])
        y_valid = np.array([self.label_encoder[str(label)] for label in valid_df[label_col]])
        y_test = np.array([self.label_encoder[str(label)] for label in test_df[label_col]])

        print(f"\nData shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_valid: {X_valid.shape}")
        print(f"  y_valid: {y_valid.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")

        return X_train, y_train, X_valid, y_valid, X_test, y_test
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        data = {
            'vocab': self.preprocessor.vocab,
            'reverse_vocab': self.preprocessor.reverse_vocab,
            'vocab_size': self.preprocessor.vocab_size,
            'max_length': self.preprocessor.max_length,
            'label_encoder': self.label_encoder,
            'reverse_label_encoder': self.reverse_label_encoder,
            'num_classes': self.num_classes
        }
        
        # Save vectorizer config if available
        if self.vectorizer is not None:
            data['vectorizer_config'] = self.vectorizer.get_config()
            data['vectorizer_vocabulary'] = self.vectorizer.get_vocabulary()
        
        np.save(filepath, data)
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        data = np.load(filepath, allow_pickle=True).item()
        
        self.preprocessor.vocab = data['vocab']
        self.preprocessor.reverse_vocab = data['reverse_vocab']
        self.preprocessor.vocab_size = data['vocab_size']
        self.preprocessor.max_length = data['max_length']
        
        self.label_encoder = data['label_encoder']
        self.reverse_label_encoder = data['reverse_label_encoder']
        self.num_classes = data['num_classes']
        
        # Restore vectorizer if config is available
        if 'vectorizer_config' in data and 'vectorizer_vocabulary' in data:
            self.vectorizer = keras.layers.TextVectorization.from_config(data['vectorizer_config'])
            self.vectorizer.set_vocabulary(data['vectorizer_vocabulary'])
    
    def get_vectorizer_for_keras_model(self):
        """Get the Keras TextVectorization layer for use in Keras models"""
        if self.vectorizer is not None:
            return self.vectorizer
        else:
            raise ValueError("Keras vectorization is not set up. Call prepare_data() first.")
    
    def convert_texts_for_scratch_model(self, texts: List[str]) -> np.ndarray:
        """Convert texts for use with from-scratch model (ensures compatibility)"""
        if self.vectorizer is not None:
            return self.vectorizer(tf.constant(texts)).numpy()
        else:
            # Fallback to custom preprocessing if vectorizer not available
            sequences = self.preprocessor.texts_to_sequences(texts)
            return self.preprocessor.pad_sequences(sequences, self.preprocessor.max_length)