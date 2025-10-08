"""
Progressive BERT-like models for educational webinar
Each model builds on the previous, showing evolution from simple to complex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# ============================================================================
# MODEL 0: Bag of Words - The Simplest Possible Baseline
# ============================================================================

class BagOfWordsClassifier(nn.Module):
    """
    The simplest possible text classifier:
    1. Count how many times each word appears
    2. Linear layer maps word counts to class scores
    
    Problems with this approach:
    - Ignores word order completely ("good not bad" = "bad not good")
    - No semantic understanding
    - Very sparse representations
    """
    
    def __init__(self, vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.classifier = nn.Linear(vocab_size, num_classes)
        print(f"Model 0 created: {vocab_size} vocab -> {num_classes} classes")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        batch_size = input_ids.size(0)
        
        # Create bag of words: count occurrences of each word
        bow = torch.zeros(batch_size, self.vocab_size, device=input_ids.device)
        for i in range(batch_size):
            for token_id in input_ids[i]:
                if token_id != 0:  # Ignore padding
                    bow[i, token_id] += 1
        
        # Classify based on word counts
        logits = self.classifier(bow)
        return logits

# ============================================================================
# MODEL 1: Word Embeddings - Dense Representations
# ============================================================================

class EmbeddingClassifier(nn.Module):
    """
    Improvement 1: Use learned dense word embeddings
    
    Instead of sparse one-hot vectors, each word gets a dense vector.
    This allows the model to learn that similar words should have similar representations.
    
    But we still just average all word embeddings - no attention yet!
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.classifier = nn.Linear(embed_dim, num_classes)
        print(f"Model 1 created: {vocab_size} vocab -> {embed_dim}d embeddings -> {num_classes} classes")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get dense embeddings for each word
        embeds = self.embeddings(input_ids)
        mask = (input_ids != 0).float().unsqueeze(-1)
        masked_embeds = embeds * mask
        seq_lens = mask.sum(dim=1)
        sentence_embed = masked_embeds.sum(dim=1) / (seq_lens + 1e-8)
        logits = self.classifier(sentence_embed)
        return logits



# ============================================================================ 
# MODEL 3: Self-Attention - The Big Leap!
# ============================================================================

class SelfAttentionClassifier(nn.Module):
    """
    The revolutionary idea: SELF-ATTENTION
    
    PREVIOUS MODELS: Used fixed query vectors
    SELF-ATTENTION: Each word can attend to every other word!
    
    KEY INSIGHT: Instead of a fixed query, let each word CREATE its own query.
    Then each word can decide how much to pay attention to every other word.
    
    This is the core innovation that makes transformers so powerful!
    
    EXPLANATION OF Q, K, V:
    - Query (Q): "What am I looking for?" 
    - Key (K): "What do I represent?"
    - Value (V): "What information do I contain?"
    
    PROCESS:
    1. Each word creates its own Query vector
    2. Each word creates its own Key vector  
    3. We compute similarity between every Query and every Key
    4. This gives us attention weights: how much should word i attend to word j?
    5. We use these weights to combine the Value vectors
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # The famous Q, K, V projections!
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        print(f"Model 3 created: Self-attention with Q, K, V projections")
        
    def attention_step_by_step(self, embeddings: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention explained step by step
        """
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Each word embedding gets transformed into Query, Key, Value
        Q = self.query_projection(embeddings)
        K = self.key_projection(embeddings)
        V = self.value_projection(embeddings)
        
        # Step 2: Compute attention scores (Q·K^T)"
        # Every query attends to every key: how similar are they?
        scores = Q @ K.transpose(-2, -1) / math.sqrt(embed_dim)

        # Don't attend to padding tokens
        mask_expanded = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        scores = scores.masked_fill(~mask_expanded.bool(), float('-inf'))
        
        # Step 4: Apply softmax to get attention weights"
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply attention weights to values"
        attended = attention_weights @ V
        
        return attended, attention_weights
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        
        # Get word embeddings
        embeds = self.embeddings(input_ids)
        
        # Apply self-attention
        mask = (input_ids != 0)
        attended, attention_weights = self.attention_step_by_step(embeds, mask)
        
        # Pool to sentence-level representation (simple average for now)
        mask_expanded = mask.float().unsqueeze(-1)
        masked_attended = attended * mask_expanded
        seq_lens = mask_expanded.sum(dim=1)
        sentence_embed = masked_attended.sum(dim=1) / (seq_lens + 1e-8)
        
        logits = self.classifier(sentence_embed)

        return logits

# ============================================================================
# MODEL 4: Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Problem with attention: It's position-agnostic!
    "not bad" and "bad not" would be processed identically.
    
    Solution: Add positional information to embeddings
    """
    
    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class AttentionWithPositionalEncoding(nn.Module):
    """
    Add positional information so the model knows about word order
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        print(f"Model 4 created: Self-attention + positional encoding")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get embeddings and add positional information
        embeds = self.embeddings(input_ids)
        embeds = self.pos_encoding(embeds)  # Now the model knows about word order!
        
        # Apply attention (same as before)
        batch_size, seq_len, embed_dim = embeds.shape
        mask = (input_ids != 0)
        
        Q = self.query_projection(embeds)
        K = self.key_projection(embeds)
        V = self.value_projection(embeds)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        mask_expanded = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        scores = scores.masked_fill(~mask_expanded.bool(), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Pool and classify
        mask_expanded = mask.float().unsqueeze(-1)
        masked_attended = attended * mask_expanded
        seq_lens = mask_expanded.sum(dim=1)
        sentence_embed = masked_attended.sum(dim=1) / (seq_lens + 1e-8)
        
        logits = self.classifier(sentence_embed)
        return logits

# ============================================================================
# MODEL 5: Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Run multiple attention heads in parallel
    Each head can focus on different types of relationships
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Create Q, K, V
        Q = self.query_projection(embeddings)
        K = self.key_projection(embeddings)
        V = self.value_projection(embeddings)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention for each head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.num_heads, seq_len, seq_len)
        scores = scores.masked_fill(~mask_expanded.bool(), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection
        output = self.output_projection(attended)
        return output

class MultiHeadAttentionClassifier(nn.Module):
    """
    Multiple attention heads can capture different relationships
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, num_classes: int = 2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.classifier = nn.Linear(embed_dim, num_classes)
        print(f"Model 5 created: Multi-head attention ({num_heads} heads)")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(input_ids)
        embeds = self.pos_encoding(embeds)
        
        mask = (input_ids != 0)
        attended = self.multi_head_attention(embeds, mask)
        
        # Pool and classify
        mask_expanded = mask.float().unsqueeze(-1)
        masked_attended = attended * mask_expanded
        seq_lens = mask_expanded.sum(dim=1)
        sentence_embed = masked_attended.sum(dim=1) / (seq_lens + 1e-8)
        
        logits = self.classifier(sentence_embed)
        return logits

# ============================================================================
# MODEL 6: Full Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Complete transformer block:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)  
    3. Feed-forward network
    4. Add & Norm again
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = None):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4  # BERT convention
            
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # BERT uses GELU
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        attended = self.attention(x, mask)
        x = self.norm1(x + attended)  # Add & Norm
        
        # Feed-forward with residual connection
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)  # Add & Norm
        
        return x

class TransformerClassifier(nn.Module):
    """
    Full transformer with multiple layers
    This is essentially BERT architecture!
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(embed_dim, num_classes)
        print(f"Model 6 created: Full transformer ({num_layers} layers, {num_heads} heads)")
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(input_ids)
        embeds = self.pos_encoding(embeds)
        
        mask = (input_ids != 0)
        
        # Pass through transformer blocks
        x = embeds
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Pool and classify  
        mask_expanded = mask.float().unsqueeze(-1)
        masked_x = x * mask_expanded
        seq_lens = mask_expanded.sum(dim=1)
        sentence_embed = masked_x.sum(dim=1) / (seq_lens + 1e-8)
        
        logits = self.classifier(sentence_embed)
        return logits


# ============================================================================
# MODEL 7: BERT-like with Multiple Heads
# ============================================================================

class BERTLikeModel(nn.Module):
    """
    BERT-like model with classification and masked language modeling heads
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 3, num_classes: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # BERT uses learned positional embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(100, embed_dim)  # Max length 100
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Task-specific heads
        self.classifier = nn.Linear(embed_dim, num_classes)  # Classification
        self.mlm_head = nn.Linear(embed_dim, vocab_size)     # Masked Language Modeling
        
        print(f"Model 7 created: BERT-like ({num_layers} layers, classification + MLM heads)")
        
    def forward(self, input_ids: torch.Tensor, task: str = 'classification') -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings (token + position)
        token_embeds = self.token_embeddings(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embeddings(positions)
        embeds = token_embeds + pos_embeds
        
        # Pass through transformer
        mask = (input_ids != 0)
        x = embeds
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        if task == 'classification':
            # Use [CLS] token (first token) for classification
            cls_embed = x[:, 0]
            logits = self.classifier(cls_embed)
            return logits
        elif task == 'mlm':
            # Predict all tokens for masked language modeling
            logits = self.mlm_head(x)
            return logits
        else:
            raise ValueError(f"Unknown task: {task}")

# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name: str, vocab_size: int, num_classes: int = 2, **kwargs):
    """Factory function to create models"""
    
    models = {
        'bow': BagOfWordsClassifier,
        'embedding': EmbeddingClassifier, 
        'self_attention': SelfAttentionClassifier,
        'positional': AttentionWithPositionalEncoding,
        'multi_head': MultiHeadAttentionClassifier,
        'transformer': TransformerClassifier,
        'bert_like': BERTLikeModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](vocab_size, num_classes=num_classes, **kwargs)