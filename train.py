"""
CausalFlow Training Script.

Trains a CausalFlowTransformer on single-cell perturbation data with:
- Flow matching loss (transport cost between control and perturbed distributions)
- MMD loss (population-level distributional fidelity, from scDFM)
- Optional GRN causal regularization

Usage:
    python train.py --data-dir /workspace/data --output-dir /workspace/output
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.causal_flow_transformer import CausalFlowTransformer, build_model
from model.grn_injection import GRNAttentionBias
from utils.grn_builder import load_dorothea, build_grn_adjacency, build_causal_attention_mask
from data.loaders import load_norman_dataset, collate_paired_batch


# ========================
# Flow Matching Loss
# ========================

class FlowMatchingLoss(nn.Module):
    """
    Conditional Flow Matching loss (from scDFM).

    Given control expression x0 and perturbed expression x1,
    we sample t ~ Uniform(0,1) and noise z ~ N(0,I),
    then compute the interpolated sample:
        xt = (1 - t) * x0 + t * x1 + noise * sigma * t * (1 - t)

    The model predicts the velocity field:
        v_t = model(xt, perturbation)  # should predict x1 - x0

    Loss = E[|| v_t - (x1 - x0) ||^2]
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        model: nn.Module,
        control: torch.Tensor,  # (batch, num_genes)
        perturbed: torch.Tensor,  # (batch, num_genes)
        perturbation_idx: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        batch_size = control.shape[0]

        # Sample interpolation parameter
        t = torch.rand(batch_size, device=control.device)

        # Sample noise
        noise = torch.randn_like(control)

        # Interpolate (with small noise for stability)
        sigma = self.sigma
        xt = (1 - t.unsqueeze(-1)) * control + t.unsqueeze(-1) * perturbed
        xt = xt + sigma * t.unsqueeze(-1) * (1 - t.unsqueeze(-1)) * noise

        # Target velocity: x1 - x0 (the optimal transport plan)
        target_v = perturbed - control

        # Predicted velocity
        pred_v = model(xt, perturbation_idx)

        # Flow matching loss
        loss = F.mse_loss(pred_v, target_v, reduction="none").mean(dim=-1)
        loss = (t.squeeze() * loss).mean()  # Weight by t

        return loss


# ========================
# MMD Loss (Population-level)
# ========================

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian kernel for MMD computation."""
    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)
    return torch.exp(-((x - y) ** 2).sum(-1) / (2 * sigma ** 2))


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy loss between two distributions."""
    loss_xy = gaussian_kernel(x, y, sigma).mean()
    loss_xx = gaussian_kernel(x, x, sigma).mean()
    loss_yy = gaussian_kernel(y, y, sigma).mean()
    return loss_xy - 0.5 * loss_xx - 0.5 * loss_yy


class MMDLoss(nn.Module):
    """Population-level MMD loss for distributional fidelity."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        model: nn.Module,
        control: torch.Tensor,
        perturbed: torch.Tensor,
        perturbation_idx: torch.Tensor,
        num_samples: int = 50,
    ) -> torch.Tensor:
        """
        Compute MMD between predicted perturbed and true perturbed distributions.
        """
        model.eval()
        with torch.no_grad():
            # Generate predictions for a subset
            batch_idx = torch.randperm(control.shape[0])[:num_samples]
            c = control[batch_idx]
            p_idx = perturbation_idx[batch_idx]

            # Predict perturbed
            pred_perturbed = model.predict_perturbed(c, p_idx, num_steps=10)

        # True perturbed for same batch
        true_perturbed = perturbed[batch_idx]

        # MMD between predicted and true
        loss = mmd_loss(pred_perturbed, true_perturbed, self.sigma)
        return loss


# ========================
# GRN Causal Regularization
# ========================

class GRNCausalRegularizer(nn.Module):
    """
    Regularizer that encourages the learned velocity field to respect
    the GRN causal structure.

    For each TF -> target edge in the GRN, we encourage:
        If TF is perturbed and has large change, target should also change.

    Loss = -corr(velocity_i, velocity_j) for all edges (i,j) in GRN
    where i is a TF and j is its target.
    """

    def __init__(self, grn_adj: np.ndarray):
        super().__init__()
        self.register_buffer(
            "grn_adj",
            torch.from_numpy(grn_adj.copy()).float()
        )

    def forward(self, velocity: torch.Tensor, perturbation_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            velocity: (batch, num_genes) predicted velocity
            perturbation_idx: (batch,) which perturbation

        Returns:
            reg_loss: scalar — penalize velocity patterns that violate GRN structure
        """
        # Compute per-gene velocity magnitude
        v_mag = velocity.abs().mean(dim=0)  # (num_genes,)

        # For each edge (i, j) in GRN, we want v_mag[j] to correlate with v_mag[i]
        # if the edge is active (non-zero in grn_adj)

        grn = self.grn_adj
        num_genes = grn.shape[0]

        # Find active edges
        edge_mask = (grn != 0)
        if edge_mask.sum() == 0:
            return torch.tensor(0.0, device=velocity.device)

        # Compute correlation encouragement
        # For activation edges (adj > 0): high v_i -> high v_j
        # For repression edges (adj < 0): high v_i -> low v_j

        activation_loss = 0.0
        repression_loss = 0.0
        n_edges = 0

        for i in range(num_genes):
            for j in range(num_genes):
                if edge_mask[i, j]:
                    sign = grn[i, j]
                    if sign > 0:
                        # Activation: encourage v_j to be proportional to v_i
                        activation_loss = activation_loss + (v_mag[j] - v_mag[i]).abs()
                    else:
                        # Repression: encourage v_j to be inversely related to v_i
                        activation_loss = activation_loss + (v_mag[j] + v_mag[i]).abs()
                    n_edges = n_edges + 1

        if n_edges == 0:
            return torch.tensor(0.0, device=velocity.device)

        reg_loss = (activation_loss + repression_loss) / n_edges
        return reg_loss


# ========================
# Training Loop
# ========================

def train_epoch(
    model,
    train_loader,
    optimizer,
    flow_loss,
    grn_reg,
    device,
    epoch: int,
    log_interval: int = 50,
) -> dict:
    model.train()
    total_loss = 0.0
    total_flow = 0.0
    total_grn = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        control = batch["control"].to(device)       # (batch, num_genes)
        perturbed = batch["perturbed"].to(device) # (batch, num_genes)
        pert_idx = batch["perturbation_idx"].to(device)

        optimizer.zero_grad()

        # Flow matching loss: predict velocity from control -> perturbed
        flow = flow_loss(model, control, perturbed, pert_idx)

        # GRN regularization
        velocity = model(control, pert_idx)
        grn_reg_loss = grn_reg(velocity, pert_idx) if grn_reg is not None else 0.0

        loss = flow + 0.01 * grn_reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_flow += flow.item()
        total_grn += grn_reg_loss.item() if grn_reg_loss != 0 else 0.0
        n_batches += 1

        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss/n_batches:.4f}",
                "flow": f"{total_flow/n_batches:.4f}",
            })

    return {
        "loss": total_loss / n_batches,
        "flow_loss": total_flow / n_batches,
        "grn_reg": total_grn / n_batches,
    }


def validate(
    model,
    val_loader,
    flow_loss,
    device,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            control = batch["control"].to(device)
            perturbed = batch["perturbed"].to(device)
            pert_idx = batch["perturbation_idx"].to(device)

            loss = flow_loss(model, control, perturbed, pert_idx)
            total_loss += loss.item()
            n_batches += 1

    return {"val_loss": total_loss / n_batches}


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # Load GRN
    print("Loading GRN prior...")
    grn_df = load_dorothea(levels=["A", "B", "C"])

    # Load dataset
    print("Loading Norman dataset...")
    adata = load_norman_dataset(max_genes=args.max_genes)

    # Get gene names
    gene_names = adata.var_names.tolist()[:args.max_genes]
    print(f"  Dataset: {adata.n_obs} cells, {len(gene_names)} genes")

    # Build GRN matrices
    print("Building GRN matrices...")
    grn_adj = build_grn_adjacency(grn_df, gene_names, mode="causal_directed")
    grn_mask = build_causal_attention_mask(grn_adj, max_seq_len=len(gene_names))

    # Count edges
    n_edges = (grn_adj != 0).sum()
    print(f"  GRN edges: {n_edges} ({n_edges/len(gene_names)**2*100:.2f}% density)")

    # Create paired train/val loaders (split by perturbation for OOD)
    from data.loaders import PairedPerturbationDataset, collate_paired_batch

    train_dataset = PairedPerturbationDataset(
        adata,
        max_genes=args.max_genes,
        sample_controls_per_perturbed=5,
    )
    val_dataset = PairedPerturbationDataset(
        adata,
        max_genes=args.max_genes,
        sample_controls_per_perturbed=5,
    )

    # Split by perturbation (OOD)
    all_pert_pairs = list(set([
        adata.obs["perturbation_name"].iloc[p[0]]
        for p in train_dataset.pairs
    ]))
    rng = np.random.default_rng(42)
    n_perts = len(all_pert_pairs)
    n_train_perts = int(n_perts * 0.8)
    pert_perms = rng.permutation(n_perts)
    train_pert_set = set([all_pert_pairs[i] for i in pert_perms[:n_train_perts]])

    pert_names_in_pairs = [
        adata.obs["perturbation_name"].iloc[p[0]]
        for p in train_dataset.pairs
    ]
    train_idx = [i for i, p in enumerate(pert_names_in_pairs) if p in train_pert_set]
    val_idx = [i for i, p in enumerate(pert_names_in_pairs) if p not in train_pert_set]

    train_sub = torch.utils.data.Subset(train_dataset, train_idx)
    val_sub = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_sub,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_paired_batch,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_paired_batch,
        num_workers=4,
        pin_memory=True,
    )

    print(f"  Train pairs: {len(train_idx)}, Val pairs: {len(val_idx)}")

    num_genes = len(gene_names)
    num_perturbations = train_dataset.n_perturbations

    # Build model
    print(f"Building model: {num_genes} genes, {num_perturbations} perturbations...")
    model = build_model(
        num_genes=num_genes,
        num_perturbations=num_perturbations,
        grn_mask=grn_mask,
        grn_adj=grn_adj,
        d_model=args.d_model,
        num_mp_layers=args.num_layers,
        grn_strategy=args.grn_strategy,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Losses
    flow_loss = FlowMatchingLoss(sigma=args.sigma).to(device)
    grn_reg = GRNCausalRegularizer(grn_adj).to(device) if args.grn_reg else None

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, flow_loss, grn_reg, device, epoch,
            log_interval=args.log_interval,
        )

        val_metrics = validate(model, val_loader, flow_loss, device)

        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['val_loss']:.4f}")

        if args.wandb:
            wandb.log({**train_metrics, **val_metrics, "epoch": epoch})

        # Save checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pt"),
            )

        # Periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"),
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CausalFlowTransformer")

    # Data
    parser.add_argument("--data-dir", type=str, default="/workspace/data")
    parser.add_argument("--output-dir", type=str, default="/workspace/output")
    parser.add_argument("--max-genes", type=int, default=2000)

    # Model
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--grn-strategy", type=str, default="attention_bias",
                        choices=["attention_bias", "message_passing", "hybrid"])
    parser.add_argument("--grn-reg", action="store_true", help="Enable GRN causal regularization")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise level for flow matching")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=10)

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="causalflow")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
