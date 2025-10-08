"""
Enhanced dataset module for BERT webinar with CSV loading and train/validation splits
"""

import torch
import numpy as np
import pandas as pd
import random
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

class TextClassificationDataset:
    """
    Enhanced dataset for text classification with CSV loading and proper splits.
    
    Supports loading from CSV files and creating train/validation splits
    for proper model evaluation.
    """
    
    def __init__(self, csv_path: str = None, min_vocab_freq: int = 2, 
                 test_size: float = 0.2, random_state: int = 42):
        """
        Initialize dataset from CSV file or use built-in data
        
        Args:
            csv_path: Path to CSV file with 'text' and 'label' columns
            min_vocab_freq: Minimum frequency for words to be included in vocabulary
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        if csv_path:
            self._load_from_csv(csv_path)
        else:
            self._create_builtin_dataset()
        
        # Create train/validation splits
        self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
            self.texts, self.labels, test_size=test_size, random_state=random_state, 
            stratify=self.labels
        )
        
        # Build vocabulary from training data only (important!)
        self.vocab = self._build_vocab(self.train_texts, min_freq=min_vocab_freq)
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Dataset loaded: {len(self.texts)} total samples")
        print(f"Train: {len(self.train_texts)} samples")
        print(f"Validation: {len(self.val_texts)} samples")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train label distribution: {Counter(self.train_labels)}")
        print(f"Val label distribution: {Counter(self.val_labels)}")
        print(f"Sample vocab: {list(self.vocab)[:15]}")
    
    def _load_from_csv(self, csv_path: str):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("CSV must have 'text' and 'label' columns")
            
            self.texts = df['text'].tolist()
            self.labels = df['label'].tolist()
            
            print(f"Loaded {len(self.texts)} samples from {csv_path}")
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Falling back to built-in dataset")
            self._create_builtin_dataset()
    
    def _create_builtin_dataset(self):
        """Create built-in dataset (fallback)"""
        # Larger, more realistic dataset
        self.positive_reviews = [
            "this movie was absolutely fantastic and amazing",
            "incredible acting with outstanding performances throughout",
            "brilliant storytelling that kept me engaged from start to finish",
            "wonderful cinematography and excellent direction",
            "masterpiece of cinema with unforgettable characters",
            "perfect blend of drama action and emotion",
            "stellar cast delivering powerful emotional scenes",
            "beautifully crafted film with stunning visual effects",
            "exceptional writing and remarkable character development",
            "truly inspiring story with meaningful messages",
            "outstanding soundtrack complementing brilliant performances", 
            "magnificent production values and superb acting",
            "compelling narrative with excellent pacing throughout",
            "extraordinary film that exceeded all my expectations",
            "phenomenal performances by the entire cast",
            "captivating storyline with incredible attention to detail",
            "remarkable direction and outstanding cinematography work",
            "absolutely loved every minute of this incredible movie",
            "brilliant performances combined with excellent storytelling",
            "wonderful experience that left me deeply moved",
            "fantastic movie with amazing character arcs",
            "incredible film that perfectly balances humor and drama",
            "outstanding production with exceptional visual storytelling",
            "masterful direction creating an unforgettable cinematic experience",
            "excellent performances bringing characters to life beautifully",
            "stunning visuals combined with powerful emotional storytelling",
            "remarkable film showcasing incredible talent and creativity",
            "wonderful movie that delivers on every single level",
            "brilliant cinematography capturing every emotional moment perfectly",
            "exceptional filmmaking with outstanding attention to detail"
        ]
        
        self.negative_reviews = [
            "terrible movie with awful acting throughout",
            "boring plot that made no sense whatsoever",
            "worst film ever made with horrible direction",
            "completely disappointing and waste of time",
            "terrible script with unconvincing performances everywhere",
            "awful cinematography and poor production values",
            "horrible movie that failed on every level",
            "disappointing storyline with weak character development",
            "terrible acting ruining what could have been good",
            "boring film with no redeeming qualities at all",
            "awful direction leading to confusing narrative structure",
            "horrible script with cringeworthy dialogue throughout",
            "disappointing movie failing to deliver promised entertainment",
            "terrible performances by otherwise talented actors",
            "boring storyline dragging on without any purpose",
            "awful film with poor editing and terrible pacing",
            "horrible movie wasting incredible potential completely",
            "disappointing direction failing to engage the audience",
            "terrible cinematography making scenes hard to follow",
            "awful production values destroying any emotional impact",
            "boring movie that put me to sleep",
            "horrible acting making characters completely unbelievable",
            "terrible film lacking any coherent story structure",
            "disappointing movie with predictable and boring plot",
            "awful direction creating confusing and messy narrative",
            "horrible script filled with meaningless dialogue",
            "terrible movie failing to connect with audience",
            "disappointing performances lacking any emotional depth",
            "boring film with absolutely no entertainment value",
            "awful movie that completely missed the mark"
        ]
        
        self.neutral_reviews = [
            "decent movie with some good moments",
            "average film that was okay to watch",
            "reasonable story with acceptable acting",
            "mediocre movie with mixed results throughout",
            "fair film with both strengths and weaknesses",
            "adequate performances in standard storyline",
            "typical movie following predictable formula",
            "ordinary film with nothing particularly special",
            "acceptable entertainment for casual viewing",
            "standard movie meeting basic expectations",
            "reasonable effort with decent production values",
            "average storyline with competent direction",
            "fair movie with some interesting moments",
            "mediocre film that was neither great nor terrible",
            "decent acting in relatively standard story",
            "acceptable movie for weekend entertainment",
            "ordinary film with predictable but watchable plot",
            "reasonable production with average performances",
            "standard movie with typical genre elements",
            "fair entertainment meeting minimal expectations"
        ]
        
        # Create full dataset
        self.texts = []
        self.labels = []
        
        # Add positive reviews (label 1)
        for review in self.positive_reviews:
            self.texts.append(review)
            self.labels.append(1)
            
        # Add negative reviews (label 0) 
        for review in self.negative_reviews:
            self.texts.append(review)
            self.labels.append(0)
            
        # Add neutral reviews (label 2) - for 3-class classification
        for review in self.neutral_reviews:
            self.texts.append(review)
            self.labels.append(2)
        
        # Shuffle dataset
        combined = list(zip(self.texts, self.labels))
        random.shuffle(combined)
        self.texts, self.labels = zip(*combined)
        self.texts, self.labels = list(self.texts), list(self.labels)
    
    def _build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from training texts only"""
        all_words = []
        for text in texts:
            # Simple tokenization - split on spaces and remove punctuation
            words = text.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '').split()
            all_words.extend(words)
        
        # Add special tokens (BERT-style)
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        
        # Add words that appear at least min_freq times
        word_counts = Counter(all_words)
        frequent_words = [word for word, count in word_counts.items() if count >= min_freq]
        vocab.extend(sorted(frequent_words))  # Sort for consistency
        
        return vocab
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices"""
        words = text.lower().replace(',', '').replace('.', '').replace('!', '').replace('?', '').split()
        return [self.word_to_idx.get(word, self.word_to_idx['[UNK]']) for word in words]
    
    def get_train_data(self, binary: bool = True):
        """
        Get training data for classification task
        
        Args:
            binary: If True, convert to binary classification (positive vs negative/neutral)
                   If False, use 3-class classification (positive/negative/neutral)
        """
        return self._process_data(self.train_texts, self.train_labels, binary)
    
    def get_val_data(self, binary: bool = True):
        """
        Get validation data for classification task
        
        Args:
            binary: If True, convert to binary classification (positive vs negative/neutral)
                   If False, use 3-class classification (positive/negative/neutral)
        """
        return self._process_data(self.val_texts, self.val_labels, binary)
    
    def get_all_data(self, binary: bool = True):
        """Get all data (for backward compatibility)"""
        return self._process_data(self.texts, self.labels, binary)
    
    def _process_data(self, texts: List[str], labels: List[int], binary: bool = True):
        """Process texts and labels"""
        texts_encoded = []
        processed_labels = []
        
        for text, label in zip(texts, labels):
            encoded = self.tokenize(text)
            texts_encoded.append(encoded)
            
            if binary:
                # Binary: positive (1) vs not-positive (0)
                binary_label = 1 if label == 1 else 0
                processed_labels.append(binary_label)
            else:
                # 3-class: keep original labels
                processed_labels.append(label)
        
        return texts_encoded, processed_labels
    
    def get_masked_lm_data(self, mask_prob: float = 0.15, split: str = 'train'):
        """
        Create masked language modeling data (for BERT pre-training style)
        
        Args:
            mask_prob: Probability of masking each token
            split: 'train', 'val', or 'all'
        """
        if split == 'train':
            texts = self.train_texts
        elif split == 'val':
            texts = self.val_texts
        else:
            texts = self.texts
        
        masked_texts = []
        original_texts = []
        
        for text in texts:
            tokens = self.tokenize(text)
            masked_tokens = tokens.copy()
            
            # Randomly mask some tokens
            for i in range(len(tokens)):
                if random.random() < mask_prob:
                    masked_tokens[i] = self.word_to_idx['[MASK]']
            
            masked_texts.append(masked_tokens)
            original_texts.append(tokens)
        
        return masked_texts, original_texts
    
    def get_sample_text(self, idx: int = 0, split: str = 'train') -> str:
        """Get original text for demonstration"""
        if split == 'train':
            return self.train_texts[idx]
        elif split == 'val':
            return self.val_texts[idx]
        else:
            return self.texts[idx]
    
    def print_sample_data(self, num_samples: int = 5, split: str = 'train'):
        """Print sample data for inspection"""
        if split == 'train':
            texts, labels = self.train_texts, self.train_labels
            print(f"\nSample training data (first {num_samples} examples):")
        elif split == 'val':
            texts, labels = self.val_texts, self.val_labels
            print(f"\nSample validation data (first {num_samples} examples):")
        else:
            texts, labels = self.texts, self.labels
            print(f"\nSample data (first {num_samples} examples):")
        
        print("-" * 60)
        
        for i in range(min(num_samples, len(texts))):
            text = texts[i]
            label = labels[i]
            encoded = self.tokenize(text)
            
            label_name = {0: "Negative", 1: "Positive", 2: "Neutral"}[label]
            
            print(f"Example {i+1}:")
            print(f"  Text: '{text}'")
            print(f"  Label: {label} ({label_name})")
            print(f"  Encoded: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
            print(f"  Length: {len(encoded)} tokens")
            print()

if __name__ == "__main__":
    # Test with CSV file
    print("Testing CSV loading:")
    dataset = TextClassificationDataset(csv_path="movie_reviews.csv")
    
    # Print sample data
    dataset.print_sample_data(num_samples=3, split='train')
    dataset.print_sample_data(num_samples=2, split='val')
    
    # Test binary classification data
    train_texts, train_labels = dataset.get_train_data(binary=True)
    val_texts, val_labels = dataset.get_val_data(binary=True)
    print(f"\nBinary classification:")
    print(f"Train: {len(train_texts)} samples, label distribution: {Counter(train_labels)}")
    print(f"Val: {len(val_texts)} samples, label distribution: {Counter(val_labels)}")
    
    # Test 3-class classification data  
    train_texts_3class, train_labels_3class = dataset.get_train_data(binary=False)
    val_texts_3class, val_labels_3class = dataset.get_val_data(binary=False)
    print(f"\n3-class classification:")
    print(f"Train: {len(train_texts_3class)} samples, label distribution: {Counter(train_labels_3class)}")
    print(f"Val: {len(val_texts_3class)} samples, label distribution: {Counter(val_labels_3class)}")