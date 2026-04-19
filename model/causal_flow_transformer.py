"""
CausalFlowTransformer: GRN-structured model for perturbation prediction.

Gene expression is a flat vector (batch, num_genes) — not a sequence.
The GRN adjacency (TF -> target edges) provides the causal structure.
We use it for GRN-informed message passing + optional attention bias.

Architecture:
    1. Per-gene embedding: each gene's expression gets a learned embedding
    2. GRN Message Passing: information flows along TF->target edges
    3. Perturbation conditioning: add perturbation embedding
    4. Flow matching head: predict velocity field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal


class GRNMessagePassingLayer(nn.Module):
    """
    One round of message passing along GRN edges.

    For each TF -> target edge in the GRN:
        message = W * h_TF (the TF's hidden state)
        target_hidden = target_hidden + message

    This is applied per-cell in the batch.
    """

    def __init__(self, d_model: int, grn_adj: np.ndarray, activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model

        # Register GRN adjacency as a buffer (not a parameter)
        self.register_buffer(
            "grn_adj",
            torch.from_numpy(grn_adj.copy()).float()
        )

        # Message function: how does TF state contribute to target
        self.message_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Normalization: divide by out-degree (each TF distributes to many targets)
        degree = grn_adj.sum(axis=1)
        degree = np.maximum(degree, 1.0)  # avoid division by zero
        self.register_buffer("degree", torch.from_numpy(degree).float())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, num_genes, d_model) — per-gene embeddings

        Returns:
            h_updated: (batch, num_genes, d_model) — after one message passing round
        """
        batch, num_genes, d_model = h.shape

        # Ensure GRN adj matches sequence length
        adj = self.grn_adj[:num_genes, :num_genes].to(h.device)
        degree = self.degree[:num_genes].to(h.device)

        # Compute messages: each gene sends its state to its targets
        messages = self.message_net(h)  # (batch, num_genes, d_model)

        # Aggregate incoming messages per target
        # adj[i,j] = 1 means gene i -> gene j (i is TF, j is target)
        # We want to aggregate messages FROM TFs TO targets
        # incoming[j] = sum_i(adj[i,j] * messages[i]) / degree[i]
        # Note: adj is TF->target, so transpose to get targets as rows

        adj_T = adj.t()  # (num_genes, num_genes) — now adj_T[j,i] = gene i -> gene j
        degree_T = degree.t()  # (num_genes,)

        # Normalize by source degree (each TF distributes equally to its targets)
        norm = degree_T.unsqueeze(1).clamp(min=1.0)  # (num_genes, 1)

        # Aggregate: for each target j, sum messages from TFs that regulate it
        # adj_T is (G, G), messages is (B, G, D), result is (B, G, D)
        incoming = torch.matmul(adj_T, messages)  # (batch, num_genes, d_model)
        incoming = incoming / norm.unsqueeze(0)  # normalize by TF's out-degree

        # Skip connection: add incoming messages to original hidden states
        h_updated = h + incoming

        return h_updated


class CausalFlowTransformer(nn.Module):
    """
    GRN-structured model for perturbation prediction with flow matching.

    The model processes a flat gene expression vector (not a sequence) by:
    1. Embedding each gene's expression value
    2. Running GRN-structured message passing layers
    3. Conditioning on the perturbation
    4. Predicting the velocity field (direction of shift toward perturbed state)
    """

    def __init__(
        self,
        num_genes: int,
        num_perturbations: int,
        grn_adj: np.ndarray,
        grn_mask: np.ndarray,
        d_model: int = 256,
        num_mp_layers: int = 4,
        dropout: float = 0.1,
        grn_strategy: str = "message_passing",
    ):
        """
        Args:
            num_genes: Number of genes
            num_perturbations: Number of unique perturbations
            grn_adj: (num_genes, num_genes) signed adjacency matrix
            grn_mask: (num_genes, num_genes) attention mask (not used in current strategy)
            d_model: Hidden dimension
            num_mp_layers: Number of GRN message passing rounds
            grn_strategy: 'message_passing' (current), 'attention_bias' (future)
        """
        super().__init__()
        self.num_genes = num_genes
        self.d_model = d_model
        self.num_mp_layers = num_mp_layers
        self.grn_strategy = grn_strategy

        # Per-gene embedding: each gene gets its own linear encoder
        # Input: scalar expression value per gene
        # Output: d_model embedding
        self.gene_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # GRN message passing layers
        self.mp_layers = nn.ModuleList([
            GRNMessagePassingLayer(d_model, grn_adj)
            for _ in range(num_mp_layers)
        ])

        # Layer norm after each MP round
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_mp_layers)
        ])

        # Perturbation embedding
        self.perturbation_embedding = nn.Embedding(num_perturbations, d_model)

        # Final prediction head
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),  # Predict velocity per gene
        )

        # Perturbation-specific gate: modulate velocity per gene
        self.perturbation_gate = nn.Embedding(num_perturbations, num_genes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        expression: torch.Tensor,  # (batch, num_genes)
        perturbation_idx: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """
        Args:
            expression: (batch, num_genes) — control gene expression (non-negative)
            perturbation_idx: (batch,) — which perturbation

        Returns:
            velocity: (batch, num_genes) — predicted velocity field
        """
        batch_size = expression.shape[0]

        # Step 1: Per-gene embedding
        # expression: (batch, num_genes) -> (batch, num_genes, 1)
        gene_input = expression.unsqueeze(-1)  # (batch, num_genes, 1)
        h = self.gene_encoder(gene_input)  # (batch, num_genes, d_model)

        # Step 2: GRN-structured message passing
        for mp_layer, ln in zip(self.mp_layers, self.layer_norms):
            h = mp_layer(h)
            h = ln(h)

        # Step 3: Perturbation conditioning
        pert_emb = self.perturbation_embedding(perturbation_idx)  # (batch, d_model)
        # Add perturbation vector to each gene's embedding
        h = h + pert_emb.unsqueeze(1)  # (batch, num_genes, d_model)

        # Step 4: Predict velocity per gene
        velocity = self.velocity_head(h).squeeze(-1)  # (batch, num_genes)

        # Step 5: Perturbation-specific gating
        gate = self.perturbation_gate(perturbation_idx)  # (batch, num_genes)
        velocity = velocity * torch.sigmoid(gate)

        return velocity

    def predict_perturbed(
        self,
        expression: torch.Tensor,
        perturbation_idx: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Predict perturbed expression via Euler integration of the velocity field.

        x_{t+1} = x_t + velocity(x_t, perturbation) * dt

        Args:
            expression: (batch, num_genes)
            perturbation_idx: (batch,)
            num_steps: Number of integration steps

        Returns:
            predicted: (batch, num_genes)
        """
        self.eval()
        with torch.no_grad():
            x = expression.clone()
            dt = 1.0 / num_steps

            for _ in range(num_steps):
                velocity = self.forward(x, perturbation_idx)
                x = x + velocity * dt
                x = F.relu(x)  # Expression is non-negative

            return x

    def causal_trace(
        self,
        expression: torch.Tensor,
        perturbation_idx: torch.Tensor,
        gene_names: list[str],
        num_steps: int = 10,
    ) -> dict:
        """
        Identify which genes are most affected by a perturbation and why.

        Returns the top changed genes and their GRN context.
        """
        self.eval()
        with torch.no_grad():
            predicted = self.predict_perturbed(expression, perturbation_idx, num_steps)
            delta = predicted - expression
            change_magnitude = delta.abs().mean(dim=0)  # (num_genes,)

            # Top changed genes
            top_indices = change_magnitude.topk(min(20, len(gene_names))).indices
            top_genes = [
                (gene_names[i] if i < len(gene_names) else f"gene_{i}",
                 change_magnitude[i].item())
                for i in top_indices
            ]

            return {
                "top_changed_genes": top_genes,
                "delta": delta.cpu().numpy(),
                "predicted": predicted.cpu().numpy(),
                "control": expression.cpu().numpy(),
            }


def build_model(
    num_genes: int,
    num_perturbations: int,
    grn_adj: np.ndarray,
    grn_mask: np.ndarray,
    d_model: int = 256,
    num_mp_layers: int = 4,
    **kwargs,
) -> CausalFlowTransformer:
    """Build CausalFlowTransformer with given config."""
    return CausalFlowTransformer(
        num_genes=num_genes,
        num_perturbations=num_perturbations,
        grn_adj=grn_adj,
        grn_mask=grn_mask,
        d_model=d_model,
        num_mp_layers=num_mp_layers,
        **kwargs,
    )
