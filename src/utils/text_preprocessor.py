"""
Data processing and text preprocessing utilities
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import re
from collections import Counter
import os


class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.max_length = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization (split by whitespace and punctuation)"""
        # Simple tokenization - can be enhanced
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_vocabulary(self, texts: List[str], max_vocab_size: int = 10000, 
                        min_freq: int = 2) -> None:
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text documents
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for word inclusion
        """
        # Count word frequencies
        word_counts = Counter()
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            word_counts.update(tokens)
        
        # Filter by minimum frequency and take top words
        filtered_words = [
            word for word, count in word_counts.items() 
            if count >= min_freq
        ]
        
        # Sort by frequency and take top words
        top_words = sorted(filtered_words, key=lambda x: word_counts[x], reverse=True)
        top_words = top_words[:max_vocab_size-2]  # Reserve space for special tokens
        
        # Build vocabulary (reserve 0 for padding, 1 for unknown)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.reverse_vocab = {0: '<PAD>', 1: '<UNK>'}
        
        for i, word in enumerate(top_words, start=2):
            self.vocab[word] = i
            self.reverse_vocab[i] = word
        
        self.vocab_size = len(self.vocab)
        
        print(f"Built vocabulary with {self.vocab_size} words")
        print(f"Most frequent words: {top_words[:10]}")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of token indices"""
        sequences = []
        
        for text in texts:
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize(cleaned_text)
            
            # Convert tokens to indices
            sequence = [
                self.vocab.get(token, self.vocab['<UNK>']) 
                for token in tokens
            ]
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], max_length: int = None, 
                     padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        """
        Pad sequences to same length
        
        Args:
            sequences: List of sequences
            max_length: Maximum length (if None, use max sequence length)
            padding: 'pre' or 'post'
            truncating: 'pre' or 'post'
            
        Returns:
            Padded sequences as numpy array
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences) if sequences else 0
        
        self.max_length = max_length
        padded_sequences = np.zeros((len(sequences), max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue
                
            # Truncate if necessary
            if len(seq) > max_length:
                if truncating == 'post':
                    seq = seq[:max_length]
                else:  # 'pre'
                    seq = seq[-max_length:]
            
            # Pad if necessary
            if len(seq) < max_length:
                if padding == 'post':
                    padded_sequences[i, :len(seq)] = seq
                else:  # 'pre'
                    padded_sequences[i, -len(seq):] = seq
            else:
                padded_sequences[i] = seq
        
        return padded_sequences
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        np.save(filepath, vocab_data)
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from file"""
        vocab_data = np.load(filepath, allow_pickle=True).item()
        self.vocab = vocab_data['vocab']
        self.reverse_vocab = vocab_data['reverse_vocab']
        self.vocab_size = vocab_data['vocab_size']
        self.max_length = vocab_data.get('max_length', None)




