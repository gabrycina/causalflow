"""
GRN Builder: Load DoRothEA/TRRUST and build adjacency matrices for GRN injection.

DoRothEA provides curated TF -> target gene causal edges with confidence levels.
Each edge has: source (TF), target (gene), mode of regulation (+/-1), confidence (A-D).
"""

import decoupler as dc
import pandas as pd
import numpy as np
from typing import Optional, Literal


def load_dorothea(
    organism: str = "human",
    levels: list = None,
    weight_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Load DoRothEA gene regulatory network.

    Note: DoRothEA in current decoupler does not include mode of regulation (mor).
    All edges are treated as activation (+1). For signed edges, use load_collectri().

    Args:
        organism: 'human' or 'mouse'
        levels: Confidence levels to include. Defaults to ['A', 'B', 'C'].
                 A = most confident, D = least confident.
        weight_scale: Scale factor for edge weights. Higher = stronger prior.

    Returns:
        DataFrame with columns: [source, target, weight, mor]
        - source: transcription factor name
        - target: target gene name
        - weight: confidence-based weight
        - mor: mode of regulation (+1 activation, -1 repression).
               Note: DoRothEA doesn't provide this, defaults to +1.
    """
    if levels is None:
        levels = ["A", "B", "C"]

    net = dc.op.dorothea(organism=organism, levels=levels)

    # Build weight from confidence level: A=1.0, B=0.75, C=0.5, D=0.25
    confidence_weights = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}
    net["weight"] = net["confidence"].map(confidence_weights) * weight_scale

    # DoRothEA doesn't have mor — default to +1 (activation) for all edges
    net["mor"] = 1.0

    return net[["source", "target", "weight", "mor"]]


def load_collectri(organism: str = "human") -> pd.DataFrame:
    """
    Load CollecTRI — expanded version of DoRothEA with sign information.

    CollecTRI includes 'sign_decision' which indicates activation vs repression.
    """
    net = dc.op.collectri(organism=organism)

    # Build weight from confidence level if available
    if "confidence" in net.columns:
        confidence_weights = {"A": 1.0, "B": 0.75, "C": 0.5, "D": 0.25}
        net["weight"] = net["confidence"].map(confidence_weights)
    else:
        net["weight"] = 1.0

    # Parse sign_decision to get mor (mode of regulation)
    # sign_decision values: "default activation", "PMID", "regulon"
    # Only "default activation" is explicitly signed; rest is treated as +1
    def parse_sign(s):
        if pd.isna(s):
            return 1.0
        s_lower = str(s).lower()
        if "repression" in s_lower or "repress" in s_lower:
            return -1.0
        return 1.0  # "default activation", "PMID", "regulon" — treat as +1

    if "sign_decision" in net.columns:
        net["mor"] = net["sign_decision"].apply(parse_sign)
    else:
        net["mor"] = 1.0

    return net[["source", "target", "weight", "mor"]]


def build_grn_adjacency(
    grn_df: pd.DataFrame,
    gene_names: list[str],
    mode: Literal["causal_directed", "undirected", "activation_only"] = "causal_directed",
) -> np.ndarray:
    """
    Build a gene x gene adjacency matrix from a GRN DataFrame.

    Args:
        grn_df: DataFrame from load_dorothea() or load_collectri()
        gene_names: Ordered list of gene names matching the expression matrix.
        mode:
            - 'causal_directed': Use directed edges TF->target, respect sign of regulation
            - 'undirected': Symmetric adjacency (interaction, not causal)
            - 'activation_only': Only positive (activating) edges

    Returns:
        adjacency_matrix: (num_genes, num_genes) numpy array.
                          Entry [i,j] = weight if gene_i -> gene_j (TF -> target)
                          Entry [i,j] = 0 if no known regulatory relationship.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n = len(gene_names)
    adj = np.zeros((n, n), dtype=np.float32)

    for _, row in grn_df.iterrows():
        tf = row["source"]
        target = row["target"]
        weight = row["weight"]
        mor = row.get("mor", 1.0)  # mode of regulation

        if tf not in gene_to_idx or target not in gene_to_idx:
            continue

        i = gene_to_idx[tf]
        j = gene_to_idx[target]

        if mode == "causal_directed":
            # Directed: TF -> target, respect activation/repression sign
            # Positive mor = activation, negative mor = repression
            adj[i, j] = weight * mor  # mor is +1 or -1
        elif mode == "undirected":
            adj[i, j] = weight
            adj[j, i] = weight
        elif mode == "activation_only":
            if mor > 0:
                adj[i, j] = weight

    return adj


def build_causal_attention_mask(
    adj: np.ndarray,
    max_seq_len: int,
    zero_diag: bool = True,
) -> np.ndarray:
    """
    Build an attention mask from a GRN adjacency matrix.

    For each gene pair (i, j): if adj[i, j] != 0, then gene i is allowed
    to attend to gene j (information can flow from TF i to target j).

    The mask is (max_seq_len, max_seq_len) and used as an additive bias:
        attention_scores = attention_scores + mask

    Args:
        adj: (num_genes, num_genes) adjacency matrix from build_grn_adjacency()
        max_seq_len: Maximum sequence length (pad/truncate to this)
        zero_diag: If True, zero out self-loops (genes don't attend to themselves)

    Returns:
        mask: (max_seq_len, max_seq_len) attention mask.
               Values: 0 = no attention allowed, 1 = allowed (weighted by edge weight)
    """
    n_genes = adj.shape[0]
    effective_len = min(n_genes, max_seq_len)

    mask = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)

    # Fill in the causal GRN structure
    grn_mask = adj[:effective_len, :effective_len].copy()

    if zero_diag:
        np.fill_diagonal(grn_mask, 0)

    # Normalize to [0, 1] for additive masking (1 = allow, 0 = block)
    # We use the absolute value to determine presence, sign is stored in adj
    grn_binary = (np.abs(grn_mask) > 0).astype(np.float32)

    # Soft weighting: multiply by normalized edge weights
    max_weight = np.max(np.abs(grn_mask))
    if max_weight > 0:
        grn_weighted = grn_binary * (np.abs(grn_mask) / max_weight)
    else:
        grn_weighted = grn_binary

    mask[:effective_len, :effective_len] = grn_weighted

    return mask


def get_grn_sign_matrix(adj: np.ndarray) -> np.ndarray:
    """
    Get the sign matrix (activation/repression) from adjacency.

    Returns:
        sign_matrix: (n, n) where +1 = activation, -1 = repression, 0 = no edge
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sign = np.where(adj != 0, np.sign(adj), 0)
    return sign
