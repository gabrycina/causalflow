"""
GRN Injection: Mechanisms for injecting GRN causal priors into a transformer.

Provides three injection strategies:
(A) Attention Bias: Add GRN adjacency as an additive bias to attention scores
(B) Message Passing: Use GRN edges for explicit message passing between layers
(C) Hybrid: Combine attention bias + message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class GRNAttentionBias(nn.Module):
    """
    Strategy (A): Inject GRN as an additive bias in attention scores.

    Before computing attention scores Q @ K^T / sqrt(d), we add the GRN
    adjacency matrix as a bias: attention = softmax((Q @ K^T + grn_bias) / sqrt(d))

    The GRN edge TF -> target means: TF can attend to target (information flows
    from TF to its targets). This encodes the causal direction.

    Optionally: use edge sign (+/-1) to boost activation or suppress repression.
    """

    def __init__(
        self,
        grn_mask: np.ndarray,
        num_heads: int = 8,
        sign_decay: float = 0.1,
        # If True, learned scaling per head; if False, uniform
        learnable_scale: bool = True,
    ):
        """
        Args:
            grn_mask: (num_genes, num_genes) attention mask from build_causal_attention_mask().
                      Values in [0, 1] indicating allowed attention edges.
            num_heads: Number of attention heads.
            sign_decay: How much to suppress negative (repression) edges vs positive.
                       Lower = more suppression of repression edges.
            learnable_scale: If True, each head has a learned scale for GRN bias.
        """
        super().__init__()
        self.num_heads = num_heads
        self.sign_decay = sign_decay

        num_genes = grn_mask.shape[0]
        self.register_buffer(
            "grn_bias",
            torch.from_numpy(grn_mask).float()
        )

        if learnable_scale:
            # Per-head learned scaling of GRN bias
            self.head_scale = nn.Parameter(torch.ones(num_heads))
        else:
            self.head_scale = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with GRN bias injected.

        Args:
            query: (batch * num_heads, seq_len, head_dim)
            key: (batch * num_heads, seq_len, head_dim)
            value: (batch * num_heads, seq_len, head_dim)
            attn_mask: Optional additive attention mask (already applied elsewhere)

        Returns:
            output: (batch * num_heads, seq_len, head_dim)
        """
        batch_heads, seq_len, head_dim = query.shape

        # Compute standard attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)

        # Inject GRN bias: broadcast grn_bias to attention size
        # grn_bias: (num_genes, num_genes) -> (1, 1, seq_len, seq_len) or (1, num_heads, seq_len, seq_len)
        grn = self.grn_bias[:seq_len, :seq_len]

        if self.head_scale is not None and self.num_heads > 1:
            # Apply per-head scaling
            grn = grn.unsqueeze(0) * self.head_scale.view(1, self.num_heads, 1, 1)

        scores = scores + grn

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class GRNMessagePassing(nn.Module):
    """
    Strategy (B): After each transformer layer, run explicit message passing
    along GRN edges only.

    Instead of (or in addition to) soft attention to all positions, information
    must explicitly flow through GRN edges: each step, node i sends its
    hidden state to all target genes j where GRN[i,j] != 0.

    This is a form of hard-wired inductive bias: the model can't route
    information outside the known causal GRN structure.
    """

    def __init__(
        self,
        grn_adj: np.ndarray,
        hidden_dim: int,
        num_passes: int = 1,
        activation: str = "gelu",
    ):
        """
        Args:
            grn_adj: (num_genes, num_genes) adjacency matrix from build_grn_adjacency().
                     Values: +1 = activation, -1 = repression, 0 = no edge.
            hidden_dim: Dimension of gene embeddings.
            num_passes: Number of message passing rounds per transformer layer.
            activation: 'gelu' or 'relu'.
        """
        super().__init__()
        self.grn_adj = torch.from_numpy(grn_adj.copy()).float()
        self.hidden_dim = hidden_dim
        self.num_passes = num_passes

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

        # Per-channel scaling of message contribution
        self.message_scale = nn.Parameter(torch.ones(hidden_dim))
        self.message_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run message passing along GRN edges.

        Args:
            x: (batch, num_genes, hidden_dim) gene embeddings

        Returns:
            x_mp: (batch, num_genes, hidden_dim) updated embeddings after message passing
        """
        batch, num_genes, hidden_dim = x.shape

        # Ensure GRN adj matches sequence length
        adj = self.grn_adj[:num_genes, :num_genes].to(x.device)

        # Normalize adjacency by out-degree (each TF distributes equally to its targets)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / degree

        x_mp = x.clone()
        for _ in range(self.num_passes):
            # Compute messages: x[i] -> all target genes j where adj[i,j] != 0
            # messages: (batch, num_genes, hidden_dim)
            messages = x_mp  # Each node sends its current state

            # Weighted aggregation: sum over incoming edges (targets receive from TFs)
            # agg[j] = sum_i(adj[i,j] * messages[i]) / degree[j]
            # adj[i,j] = +1 (activation) or -1 (repression) or 0
            incoming = torch.matmul(adj_norm.transpose(-2, -1), messages)

            # Apply sign of regulation: positive mor boosts, negative suppresses
            # Repression: flip the message sign before adding
            # We do: message * sign, which means repression edges subtract
            # sign matrix: (1, num_genes, 1) broadcast
            sign = torch.sign(adj_norm.sum(dim=-1, keepdim=True).clamp(min=1.0))
            # Actually: we need incoming sign per edge, not aggregated
            # Simpler: just use the adjacency sign directly
            # incoming has already summed over TF->target edges, sign is embedded

            x_mp = x_mp + self.activation(incoming * self.message_scale + self.message_bias)

        return x_mp


class GRNInjector(nn.Module):
    """
    Unified GRN injection module supporting strategies (A), (B), and (C).

    This is the main class to use in the transformer.
    """

    def __init__(
        self,
        grn_mask: np.ndarray,
        grn_adj: np.ndarray,
        hidden_dim: int,
        num_heads: int = 8,
        strategy: str = "attention_bias",
        # Attention bias params
        sign_decay: float = 0.1,
        # Message passing params
        num_mp_passes: int = 1,
    ):
        """
        Args:
            grn_mask: (num_genes, num_genes) attention mask from build_causal_attention_mask()
            grn_adj: (num_genes, num_genes) adjacency from build_grn_adjacency()
            hidden_dim: Model hidden dimension
            num_heads: Number of attention heads
            strategy: 'attention_bias', 'message_passing', or 'hybrid'
        """
        super().__init__()
        self.strategy = strategy
        self.hidden_dim = hidden_dim

        if strategy in ["attention_bias", "hybrid"]:
            self.attn_bias = GRNAttentionBias(
                grn_mask=grn_mask,
                num_heads=num_heads,
                sign_decay=sign_decay,
            )

        if strategy in ["message_passing", "hybrid"]:
            self.message_pass = GRNMessagePassing(
                grn_adj=grn_adj,
                hidden_dim=hidden_dim,
                num_passes=num_mp_passes,
            )

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attn_bias: Optional pre-computed additive attention bias from elsewhere

        Returns:
            x: Updated embeddings
            info: Dict with intermediate results (attention weights, etc.)
        """
        info = {}

        if self.strategy in ["attention_bias", "hybrid"]:
            # GRN bias is applied within the attention computation
            # The actual injection happens in the transformer's attention layer
            info["grn_injected"] = True
            info["strategy"] = self.strategy

        if self.strategy in ["message_passing", "hybrid"]:
            x = self.message_pass(x)
            info["message_passing_applied"] = True

        return x, info
