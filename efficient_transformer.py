import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any, Union

class EfficientAttention(nn.Module):
    """
    Memory-efficient implementation of multi-head attention with support for gradient checkpointing
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attention = use_flash_attention
        
        # Single projection for q, k, v for better efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters with Glorot / fan_avg
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)
    
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape from [batch_size, seq_len, embed_dim] to [batch_size, num_heads, seq_len, head_dim]"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
    
    def _scaled_dot_product_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention
        
        Args:
            q, k, v: Query, key, and value tensors [batch_size, num_heads, seq_len, head_dim]
            mask: Optional attention mask [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
            
        Returns:
            output: Attention output [batch_size, num_heads, seq_len, head_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def _flash_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flash Attention implementation for memory efficiency
        Only used when PyTorch version supports it and use_flash_attention is True
        """
        # Check if we can use Flash Attention (requires PyTorch 2.0+ and GPU)
        if hasattr(F, 'scaled_dot_product_attention') and self.use_flash_attention:
            # Use built-in flash attention when available
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            # Create dummy attention weights (not returned by flash attention)
            # If needed for visualization, implement a slower path to get these
            attn_weights = torch.zeros(q.size(0), q.size(1), q.size(2), k.size(2), device=q.device)
            return output, attn_weights
        else:
            # Fall back to regular attention
            return self._scaled_dot_product_attention(q, k, v, mask)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: Optional[torch.Tensor] = None, 
        v: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for efficient attention

        Args:
            q: Query tensor [batch_size, seq_len, embed_dim]
            k, v: Optional key and value tensors (if None, use q for self-attention)
            mask: Optional attention mask [batch_size, seq_len, seq_len]

        Returns:
            output: Attention output [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights
        """
        batch_size, seq_len, _ = q.size()
        
        # Handle self-attention case
        if k is None and v is None:
            # Combined projection for q, k, v
            qkv = self.qkv_proj(q)
            
            # Split into q, k, v
            qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
            q_proj, k_proj, v_proj = qkv[0], qkv[1], qkv[2]
        else:
            # Cross-attention case
            k = k if k is not None else q
            v = v if v is not None else q
            
            q_proj = self._reshape_for_attention(self.qkv_proj(q)[:, :, :self.embed_dim])
            k_proj = self._reshape_for_attention(self.qkv_proj(k)[:, :, self.embed_dim:2*self.embed_dim])
            v_proj = self._reshape_for_attention(self.qkv_proj(v)[:, :, 2*self.embed_dim:])
        
        # Compute attention
        if self.use_flash_attention:
            output, attention_weights = self._flash_attention(q_proj, k_proj, v_proj, mask)
        else:
            output, attention_weights = self._scaled_dot_product_attention(q_proj, k_proj, v_proj, mask)
        
        # Reshape output back to [batch_size, seq_len, embed_dim]
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, attention_weights

class MemoryEfficientFeedForward(nn.Module):
    """
    Memory-efficient feed-forward network with activation checkpointing support
    """
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: int, 
        activation: str = 'gelu',
        dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the forward pass (for checkpointing)"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing"""
        if use_checkpoint and x.requires_grad:
            x = torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            x = self._forward_impl(x)
        return x

class EfficientTransformerLayer(nn.Module):
    """
    Efficient Transformer Encoder Layer with memory optimizations
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_flash_attention: bool = False,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        self.self_attn = EfficientAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        self.feed_forward = MemoryEfficientFeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            activation=activation,
            dropout=dropout
        )
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional gradient checkpointing

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            use_checkpoint: Whether to use gradient checkpointing

        Returns:
            output: Layer output [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights
        """
        # Self attention with pre-norm
        residual = x
        x = self.norm1(x)
        
        if use_checkpoint and x.requires_grad:
            # Attention with checkpointing
            def attn_func(x_inner):
                return self.self_attn(q=x_inner, mask=mask)[0]
            
            x = torch.utils.checkpoint.checkpoint(attn_func, x)
            # Dummy attention weights (not used with checkpointing)
            attention_weights = None
        else:
            # Regular attention
            x, attention_weights = self.self_attn(q=x, mask=mask)
            
        x = residual + self.dropout(x)
        
        # Feed forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x, use_checkpoint=use_checkpoint))
        
        return x, attention_weights

class ScalableTransformerEncoder(nn.Module):
    """
    Scalable Transformer Encoder with memory efficiency optimizations
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int = 1,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_flash_attention: bool = False,
        use_mixed_precision: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_mixed_precision = use_mixed_precision
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EfficientTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                use_flash_attention=use_flash_attention
            )
            for _ in range(num_layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
            
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with mixed precision and gradient checkpointing options

        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            mask: Optional attention mask
            output_attentions: Whether to return attention weights
            use_checkpoint: Whether to use gradient checkpointing

        Returns:
            output: Encoder output [batch_size, seq_len, embed_dim]
            attentions: Optional list of attention weights from each layer
        """
        # Handle inputs with dimensions [batch_size, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
            
        # Mixed precision context
        context_manager = torch.cuda.amp.autocast() if self.use_mixed_precision else torch.no_grad()
        
        with context_manager:
            # Apply input embedding
            x = self.input_embedding(x)
            
            # Store attention weights if needed
            attentions = [] if output_attentions else None
            
            # Process through transformer layers
            for layer in self.layers:
                x, attention_weights = layer(x, mask=mask, use_checkpoint=use_checkpoint)
                if output_attentions:
                    attentions.append(attention_weights)
        
        return x, attentions

class ScalableTransformerModel(nn.Module):
    """
    Complete scalable transformer model with classification head
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_flash_attention: bool = False,
        use_mixed_precision: bool = False
    ):
        super().__init__()
        
        self.encoder = ScalableTransformerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation,
            use_flash_attention=use_flash_attention,
            use_mixed_precision=use_mixed_precision
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        use_checkpoint: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass for classification

        Args:
            x: Input tensor
            mask: Optional attention mask
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            use_checkpoint: Whether to use gradient checkpointing

        Returns:
            logits: Classification logits
            hidden_states: Hidden representation
            attentions: Optional attention weights
        """
        # Get encoder output
        encoder_output, attentions = self.encoder(
            x, 
            mask=mask, 
            output_attentions=output_attentions,
            use_checkpoint=use_checkpoint
        )
        
        # Use the representation of the first token for classification
        hidden_states = encoder_output.squeeze(1) if encoder_output.size(1) == 1 else encoder_output[:, 0]
        
        # Classification
        logits = self.classifier(hidden_states)
        
        if output_hidden_states:
            return logits, hidden_states, attentions
        else:
            return logits, None, attentions


# Additional Functionality for Mixed Precision Training

class MixedPrecisionTrainer:
    """
    Helper class for mixed precision training
    """
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        use_mixed_precision: bool = True,
        use_checkpointing: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.use_checkpointing = use_checkpointing
        
        # Initialize grad scaler for mixed precision
        self.scaler = scaler if scaler is not None and self.use_mixed_precision else None
        if self.use_mixed_precision and self.scaler is None:
            self.scaler = torch.cuda.amp.GradScaler()
            
    def training_step(self, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Perform a single training step with mixed precision support"""
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with autocast
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs, _, _ = self.model(inputs, use_checkpoint=self.use_checkpointing)
                loss = criterion(outputs, targets)
                
            # Backward pass with scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            outputs, _, _ = self.model(inputs, use_checkpoint=self.use_checkpointing)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
        return loss.item()