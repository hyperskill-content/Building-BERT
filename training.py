"""
Enhanced training utilities for BERT webinar with proper train/validation evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import numpy as np

class TextDataset(Dataset):
    """PyTorch Dataset for text classification"""
    
    def __init__(self, texts_encoded: List[List[int]], labels: List[int], max_len: int = None):
        self.texts = self._pad_sequences(texts_encoded, max_len)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def _pad_sequences(self, sequences: List[List[int]], max_len: int = None) -> torch.Tensor:
        """Pad sequences to same length"""
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padded.append(seq + [0] * (max_len - len(seq)))
            else:
                padded.append(seq[:max_len])
        
        return torch.tensor(padded, dtype=torch.long)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def create_dataloaders(train_texts: List[List[int]], train_labels: List[int],
                      val_texts: List[List[int]], val_labels: List[int],
                      batch_size: int = 8, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation"""
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """Train model for one epoch"""
    model.train()
    
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_texts, batch_labels in dataloader:
        batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(batch_texts)
        loss = criterion(logits, batch_labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    avg_loss = epoch_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                  device: str = 'cpu') -> Tuple[float, float]:
    """Evaluate model performance"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_texts, batch_labels in dataloader:
            batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
            
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

def train_model_simple(model, train_loader, val_loader, num_epochs=6, lr=1e-3):
    """Simple training with train/val tracking"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training for {num_epochs} epochs with lr={lr}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_texts, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_texts)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch_texts, batch_labels in val_loader:
                logits = model(batch_texts)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        # Print epoch results
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        
        print(f"Epoch {epoch}: Train Loss={train_loss_avg:.4f}, Train Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss_avg:.4f}, Val Acc={val_acc:.1f}%")
    
    return {
        'train_loss': train_loss_avg,
        'train_acc': train_acc,
        'val_loss': val_loss_avg,
        'val_acc': val_acc
    }
