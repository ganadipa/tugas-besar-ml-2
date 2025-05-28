from utils.text_preprocessor import TextPreprocessor
import numpy as np
import pandas as pd
import os
from typing import List, Tuple

class DataLoader:
    """Data loading and preprocessing pipeline"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.preprocessor = TextPreprocessor()
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.num_classes = 0
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, validation, and test data
        
        Returns:
            Tuple of (train_df, valid_df, test_df)
        """
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
    
    def prepare_data(self, max_vocab_size: int = 10000, max_length: int = 100,
                    min_freq: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                               np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            max_vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            min_freq: Minimum word frequency
            
        Returns:
            Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
        """
        # Load data
        train_df, valid_df, test_df = self.load_data()
        
        # Build vocabulary from training data
        train_texts = train_df['text'].astype(str).tolist()
        self.preprocessor.build_vocabulary(train_texts, max_vocab_size, min_freq)
        
        # Convert texts to sequences
        X_train = self._process_texts(train_texts, max_length)
        X_valid = self._process_texts(valid_df['text'].astype(str).tolist(), max_length)
        X_test = self._process_texts(test_df['text'].astype(str).tolist(), max_length)
        
        # Encode labels (assuming labels are in 'label' column, adjust if different)
        if 'label' in train_df.columns:
            label_col = 'label'
        elif 'sentiment' in train_df.columns:
            label_col = 'sentiment'
        else:
            # Use the first non-text column as label
            label_col = [col for col in train_df.columns if col not in ['id', 'text']][0]
        
        all_labels = (train_df[label_col].astype(str).tolist() + 
                     valid_df[label_col].astype(str).tolist() + 
                     test_df[label_col].astype(str).tolist())
        
        self.encode_labels(list(set(all_labels)))
        
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
    
    def _process_texts(self, texts: List[str], max_length: int) -> np.ndarray:
        """Process texts to padded sequences"""
        sequences = self.preprocessor.texts_to_sequences(texts)
        padded_sequences = self.preprocessor.pad_sequences(sequences, max_length)
        return padded_sequences
    
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
