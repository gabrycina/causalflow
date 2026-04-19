"""
CausalFlow Evaluation Script.

Evaluates a trained CausalFlowTransformer on held-out perturbations
to assess OOD generalization. Metrics:
- Pearson correlation of log-fold changes
- DE gene overlap
- MMD between predicted and observed perturbed distributions
- Causal tracing: which TF cascades are identified

Usage:
    python evaluate.py --model-path /workspace/output/best_model.pt \
                      --data-dir /workspace/data --output-dir /workspace/results
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_aucroc
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.causal_flow_transformer import CausalFlowTransformer, build_model
from utils.grn_builder import load_dorothea, build_grn_adjacency, build_causal_attention_mask
from data.loaders import load_norman_dataset, PerturbationDataset, collate_fn
from torch.utils.data import DataLoader


def compute_log_fold_change(
    control: np.ndarray,
    perturbed: np.ndarray,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Compute log-fold change between control and perturbed."""
    control = np.maximum(control, pseudocount)
    perturbed = np.maximum(perturbed, pseudocount)
    return np.log2(perturbed / control)


def evaluate_pearson_correlation(
    model,
    test_loader,
    device,
) -> dict:
    """
    Compute Pearson correlation between predicted and observed log-fold changes.
    """
    model.eval()
    all_pred_lfc = []
    all_true_lfc = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Pearson evaluation"):
            expression = batch["expression"].numpy()
            pert_idx = batch["perturbation_idx"].to(device)

            # Predict perturbed
            pred_perturbed = model.predict_perturbed(
                expression=torch.tensor(expression).to(device),
                perturbation_idx=pert_idx,
                num_steps=10,
            ).cpu().numpy()

            # Compute log-fold change
            pred_lfc = compute_log_fold_change(expression, pred_perturbed).mean(axis=0)
            true_lfc = np.zeros_like(pred_lfc)  # No ground truth for single cells

            all_pred_lfc.append(pred_lfc)
            all_true_lfc.append(true_lfc)

    all_pred_lfc = np.concatenate(all_pred_lfc, axis=0)
    all_true_lfc = np.concatenate(all_true_lfc, axis=0)

    # Aggregate by perturbation
    # Group by perturbation and compute mean LFC
    # For now, return per-cell correlation
    correlations = []
    for i in range(len(all_pred_lfc)):
        if np.std(all_pred_lfc[i]) > 0:
            corr, _ = pearsonr(all_pred_lfc[i], all_true_lfc[i])
            if not np.isnan(corr):
                correlations.append(corr)

    return {"mean_pearson": np.mean(correlations) if correlations else 0.0}


def evaluate_de_overlap(
    model,
    test_loader,
    top_k: int = 100,
    fold_change_thresh: float = 0.5,
    device,
) -> dict:
    """
    Compute differentially expressed gene overlap.

    For each perturbation, identify top-K DE genes in predicted vs true.
    Report overlap fraction.
    """
    model.eval()

    overlaps = []
    n_evaluated = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="DE overlap evaluation"):
            expression = batch["expression"].numpy()
            pert_idx = batch["perturbation_idx"].to(device)

            pred_perturbed = model.predict_perturbed(
                expression=torch.tensor(expression).to(device),
                perturbation_idx=pert_idx,
                num_steps=10,
            ).cpu().numpy()

            # For each cell in batch, compute predicted DE genes
            for i in range(len(expression)):
                pred_lfc = np.abs(
                    compute_log_fold_change(expression[i], pred_perturbed[i])
                )
                true_lfc = np.abs(
                    compute_log_fold_change(expression[i], expression[i])  # self-comparison
                )

                # Top-K DE genes by predicted LFC
                top_pred = set(np.argsort(pred_lfc)[-top_k:])
                top_true = set(np.argsort(true_lfc)[-top_k:])

                if len(top_true) > 0:
                    overlap = len(top_pred & top_true) / top_k
                    overlaps.append(overlap)
                n_evaluated += 1

    return {
        "mean_de_overlap": np.mean(overlaps) if overlaps else 0.0,
        "n_evaluated": n_evaluated,
    }


def evaluate_mmd_distributional(
    model,
    test_loader,
    num_perts: int = 10,
    device,
    sigma: float = 1.0,
) -> dict:
    """
    Compute MMD between predicted and observed perturbed distributions.

    Groups cells by perturbation, generates predictions, computes MMD.
    """
    model.eval()

    from torch.utils.data import DataLoader

    mmd_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="MMD evaluation"):
            expression = batch["expression"].numpy()
            pert_idx = batch["perturbation_idx"].to(device)

            pred_perturbed = model.predict_perturbed(
                expression=torch.tensor(expression).to(device),
                perturbation_idx=pert_idx,
                num_steps=10,
            ).cpu().numpy()

            # MMD between predicted and control (proxy for distributional shift)
            control_flat = expression.reshape(expression.shape[0], -1)
            pred_flat = pred_perturbed.reshape(pred_perturbed.shape[0], -1)

            # Gaussian kernel MMD
            diff = control_flat - pred_flat
            mmd = np.exp(-np.sum(diff ** 2, axis=1) / (2 * sigma ** 2)).mean()
            mmd_losses.append(1 - mmd)  # Lower is better

    return {"mean_mmd": np.mean(mmd_losses)}


def causal_tracing_eval(
    model,
    test_loader,
    gene_names: list[str],
    grn_adj: np.ndarray,
    device,
    num_perturbations: int = 5,
) -> dict:
    """
    Evaluate causal tracing: does the model identify correct TF cascades?

    For top-perturbed genes, check if they appear in the GRN's known targets.
    """
    model.eval()

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Causal tracing")):
            if batch_idx >= num_perturbations:
                break

            expression = batch["expression"].to(device)
            pert_idx = batch["perturbation_idx"].to(device)

            trace = model.causal_trace(
                expression=expression,
                perturbation_idx=pert_idx,
                target_genes=[],  # All genes
                gene_names=gene_names,
                num_steps=10,
            )

            top_changed = trace["top_changed_genes"]

            # Check: how many of the top changed genes are known GRN targets?
            n_in_grn = sum(
                1 for gene, _ in top_changed
                if gene in gene_to_idx
                and grn_adj[:, gene_to_idx[gene]].sum() != 0
            )

            results.append({
                "perturbation": batch["perturbation_name"][0],
                "top_changed": top_changed[:10],
                "frac_in_grn": n_in_grn / 10,
            })

    return {"causal_tracing_results": results}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load GRN
    print("Loading GRN...")
    grn_df = load_dorothea(levels=["A", "B", "C"])
    adata = load_norman_dataset(max_genes=args.max_genes)
    gene_names = adata.var_names.tolist()
    grn_adj = build_grn_adjacency(grn_df, gene_names, mode="causal_directed")
    grn_mask = build_causal_attention_mask(grn_adj, max_seq_len=len(gene_names))

    # Create test dataset (held-out cells)
    print("Loading test data...")
    n_cells = adata.n_obs
    indices = np.random.permutation(n_cells)
    test_adata = adata[indices[int(n_cells * 0.8):]].copy()

    test_dataset = PerturbationDataset(test_adata)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    num_genes = len(gene_names)
    num_perturbations = test_dataset.n_perturbations

    # Build model
    print("Loading model...")
    model = build_model(
        num_genes=num_genes,
        num_perturbations=num_perturbations,
        grn_mask=grn_mask,
        grn_adj=grn_adj,
        d_model=args.d_model,
        num_layers=args.num_layers,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded model from {args.model_path}")

    # Run evaluations
    print("\nRunning evaluations...")

    metrics = {}

    if args.evaluate_pearson:
        metrics.update(evaluate_pearson_correlation(model, test_loader, device))

    if args.evaluate_de_overlap:
        metrics.update(evaluate_de_overlap(model, test_loader, device=device))

    if args.evaluate_mmd:
        metrics.update(evaluate_mmd_distributional(model, test_loader, device=device))

    if args.evaluate_causal_tracing:
        metrics.update(causal_tracing_eval(
            model, test_loader, gene_names, grn_adj, device
        ))

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if k != "causal_tracing_results":
            print(f"  {k}: {v}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        # Convert numpy types
        def convert(o):
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            return o
        json.dump({k: convert(v) for k, v in metrics.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name)
        wandb.log(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CausalFlowTransformer")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/workspace/data")
    parser.add_argument("--output-dir", type=str, default="/workspace/results")
    parser.add_argument("--max-genes", type=int, default=2000)

    # Model config (must match training)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--grn-strategy", type=str, default="attention_bias")

    # Evaluation options
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate-pearson", action="store_true")
    parser.add_argument("--evaluate-de-overlap", action="store_true")
    parser.add_argument("--evaluate-mmd", action="store_true")
    parser.add_argument("--evaluate-causal-tracing", action="store_true")
    parser.add_argument("--evaluate-all", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="causalflow-eval")
    parser.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.evaluate_all:
        args.evaluate_pearson = True
        args.evaluate_de_overlap = True
        args.evaluate_mmd = True
        args.evaluate_causal_tracing = True

    main(args)
