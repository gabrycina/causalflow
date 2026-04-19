"""
Data loaders for Norman and Adamson perturbation datasets.

Handles the control-perturbed pairing problem:
For each perturbation, we need to identify the matched control cells
from the SAME experimental batch (gemgroup), not just any control.

Key insight from Norman et al. 2019:
- Each cell is assigned to a "gemgroup" — an experimental batch
- Within each gemgroup, there are both control cells (non-targeting guide)
  and perturbed cells (gene-targeting guide)
- For training: pair each perturbed cell with control cells from the same gemgroup
"""

import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Literal
import warnings

warnings.filterwarnings("ignore")


def load_norman_dataset(
    data_dir: str = "/workspace/data",
    max_genes: int = 2000,
    organism: str = "human",
) -> "anndata.AnnData":
    """
    Load the Norman et al. 2019 Perturb-seq dataset.

    The Norman dataset is a single-cell Perturb-seq dataset measuring
    transcriptional responses to CRISPR perturbations in K562 cells.

    Data is loaded from local h5ad file if available, otherwise downloaded
    from the scVI-data repository or built from scratch.

    Args:
        data_dir: Directory containing data files
        max_genes: Maximum number of genes to keep (by variance)
        organism: 'human' or 'mouse'

    Returns:
        AnnData object with the Norman perturbation data
    """
    import anndata as ad
    import scanpy as sc

    # Check for local file first - try both naming conventions
    local_paths = [
        os.path.join(data_dir, "NormanWeissman2019_filtered.h5ad"),
        os.path.join(data_dir, "norman_2019.h5ad"),
    ]
    local_path = None
    for p in local_paths:
        if os.path.exists(p):
            local_path = p
            break

    if local_path:
        print(f"Loading Norman dataset from {local_path}")
        adata = ad.read_h5ad(local_path)
    else:
        # Try to load via scvi-tools if available
        try:
            import scvi

            print("Loading Norman dataset via scVI-tools...")
            # scvi.data reads the Norman data from a URL or cache
            adata = scvi.data.read_perturbation_data(
                dataset="norman2019",
                organism=organism,
            )
        except Exception as e:
            # Fallback: create a minimal synthetic dataset for testing
            # This allows the code to run even without the real data
            print(f"Warning: Could not load Norman dataset ({e})")
            print("Creating synthetic test dataset...")
            adata = create_synthetic_perturbation_data(
                n_cells=5000,
                n_genes=5000,
                n_perturbations=20,
                n_gemgroups=10,
            )

    # Preprocess
    if max_genes < adata.n_vars:
        print(f"Selecting top {max_genes} highly variable genes...")
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes, flavor="seurat_v3")
        adata = adata[:, adata.var.highly_variable]

    # Log1p normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Ensure perturbation_name and gemgroup columns exist
    if "perturbation_name" not in adata.obs.columns:
        if "perturbation" in adata.obs.columns:
            adata.obs["perturbation_name"] = adata.obs["perturbation"]
        elif "guide" in adata.obs.columns:
            adata.obs["perturbation_name"] = adata.obs["guide"]
        else:
            adata.obs["perturbation_name"] = "unknown"

    if "gemgroup" not in adata.obs.columns:
        adata.obs["gemgroup"] = 0

    print(f"  Dataset: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"  Perturbations: {adata.obs['perturbation_name'].nunique()}")
    print(f"  Gemgroups: {adata.obs['gemgroup'].nunique()}")

    return adata


def create_synthetic_perturbation_data(
    n_cells: int = 5000,
    n_genes: int = 5000,
    n_perturbations: int = 20,
    n_gemgroups: int = 10,
) -> "anndata.AnnData":
    """
    Create a synthetic perturbation dataset for testing.

    This mimics the structure of real Perturb-seq data with:
    - Multiple gemgroups (experimental batches)
    - Control cells and perturbed cells
    - Binary perturbation effects on specific genes
    """
    import anndata as ad

    np.random.seed(42)

    # Create gene names
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Create perturbations
    perturbations = [f"pert_{i}" for i in range(n_perturbations)]
    control_perts = ["NON_TARGETING"] * 3  # Some control perturbations

    all_perts = control_perts + perturbations

    # Create cells
    cells_per_gemgroup = n_cells // n_gemgroups
    obs_data = []
    X_data = []

    for gem in range(n_gemgroups):
        base_expression = np.random.randn(n_genes).astype(np.float32) * 0.5 + 5
        base_expression = np.clip(base_expression, 0, None)

        for cell_idx in range(cells_per_gemgroup):
            # 20% chance of being a control cell
            is_control = np.random.rand() < 0.2

            if is_control:
                pert_name = np.random.choice(control_perts)
                expression = base_expression + np.random.randn(n_genes).astype(np.float32) * 0.3
            else:
                pert_name = np.random.choice(perturbations)
                # Apply perturbation effect: upregulate some random genes
                effect = np.zeros(n_genes, dtype=np.float32)
                n_effected_genes = np.random.randint(5, 20)
                effected = np.random.choice(n_genes, n_effected_genes, replace=False)
                effect[effected] = np.random.randn(n_effected_genes).astype(np.float32) * 2
                expression = base_expression + effect + np.random.randn(n_genes).astype(np.float32) * 0.3

            expression = np.clip(expression, 0, None)
            obs_data.append({
                "perturbation_name": pert_name,
                "gemgroup": gem,
                "is_control": pert_name in control_perts,
            })
            X_data.append(expression)

    obs = pd.DataFrame(obs_data)
    X = np.array(X_data, dtype=np.float32)

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_names

    return adata


def identify_control_pattern(adata) -> pd.Series:
    """
    Identify which cells are controls (non-targeting guides) vs perturbed.

    Returns a boolean Series: True = control cell.
    """
    obs = adata.obs

    if "perturbation_name" in obs.columns:
        # Norman dataset: control = non-targeting or empty perturbation
        # Convert to object type first to avoid Categorical fillna issues
        pert = obs["perturbation_name"].astype(object).fillna("unknown")
        is_control = pert.str.contains(
            "NON-TARGET|NT|Control|control|non-target|NONTCELL",
            case=False,
            na=False
        )
        return is_control

    # Fallback: look for guide columns
    guide_cols = [c for c in obs.columns if "guide" in c.lower() and "identity" in c.lower()]
    if guide_cols:
        guide_col = guide_cols[0]
        is_control = obs[guide_col].str.contains(
            "NON-TARGET|NT|control|NONTCELL",
            case=False,
            na=False
        )
        return is_control

    # Last resort: assume first perturbation_name is control
    raise ValueError("Cannot identify control cells in dataset. Need 'perturbation_name' or guide_identity column.")


class PairedPerturbationDataset(Dataset):
    """
    Dataset that properly pairs perturbed cells with matched control cells.

    For each perturbation (and gemgroup), we:
    1. Find the perturbed cells
    2. Find control cells from the SAME gemgroup
    3. For training: for each perturbed cell, sample matched controls

    The model sees: (perturbed_cell, control_cell, perturbation_idx)
    and learns to predict: perturbed_cell from control_cell + perturbation
    """

    def __init__(
        self,
        adata,
        perturbation_col: str = "perturbation_name",
        gemgroup_col: str = "gemgroup",
        max_genes: int = 2000,
        sample_controls_per_perturbed: int = 5,
        include_control_cells: bool = False,
        random_seed: int = 42,
    ):
        """
        Args:
            adata: AnnData object with perturbation data
            perturbation_col: Column with perturbation names
            gemgroup_col: Column with experimental batch ID
            max_genes: Max genes to include (by variance)
            sample_controls_per_perturbed: How many control cells to pair with each perturbed cell
            include_control_cells: If True, also predict perturbation effects on control cells
        """
        self.adata = adata
        self.perturbation_col = perturbation_col
        self.gemgroup_col = gemgroup_col
        self.max_genes = max_genes
        self.sample_controls = sample_controls_per_perturbed
        self.include_control = include_control_cells
        self.rng = np.random.default_rng(random_seed)

        obs = adata.obs.copy()

        # Identify controls vs perturbed
        self.is_control = identify_control_pattern(adata)

        # Get gene list
        if "highly_variable" in adata.var.columns:
            hvg = adata.var_names[adata.var.highly_variable].tolist()
            self.gene_names = hvg[:max_genes]
        else:
            self.gene_names = adata.var_names.tolist()[:max_genes]
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

        # Build (perturbed_cell, matched_control_cells) pairs
        self.pairs = self._build_pairs(obs)

        # Build perturbation encoder
        pert_names = obs[perturbation_col].unique()
        pert_names = [p for p in pert_names if pd.notna(p)]
        self.perturbation_to_idx = {p: i for i, p in enumerate(sorted(pert_names))}
        self.n_perturbations = len(self.perturbation_to_idx)

    def _build_pairs(self, obs: pd.DataFrame) -> list:
        """
        Build list of (perturbed_idx, control_idx_list) pairs.

        For each perturbation in each gemgroup, find matched controls.
        """
        pairs = []

        perts = obs[self.perturbation_col].astype(object).fillna("unknown")
        gems = obs[self.gemgroup_col] if self.gemgroup_col in obs.columns else pd.Series(0, index=obs.index)

        # Group cells by (gemgroup, perturbation)
        for gem in gems.unique():
            gem_mask = gems == gem
            for pert in obs[gem_mask][self.perturbation_col].unique():
                if pd.isna(pert) or pert == "unknown":
                    continue

                pert_mask = gem_mask & (perts == pert)
                ctrl_mask = gem_mask & self.is_control

                pert_indices = np.where(pert_mask)[0]
                ctrl_indices = np.where(ctrl_mask)[0]

                if len(ctrl_indices) == 0:
                    # No matched controls — use global controls as fallback
                    ctrl_indices = np.where(self.is_control)[0]

                if len(pert_indices) == 0:
                    continue

                # For each perturbed cell, store indices of matched controls
                for pert_idx in pert_indices:
                    # Sample a subset of controls (or all if fewer)
                    n_sample = min(self.sample_controls, len(ctrl_indices))
                    sampled_ctrl = self.rng.choice(ctrl_indices, n_sample, replace=False)
                    pairs.append((pert_idx, sampled_ctrl.tolist()))

        # If include_control_cells, also add control cells as "perturbed by themselves"
        if self.include_control:
            ctrl_indices = np.where(self.is_control)[0]
            for ctrl_idx in ctrl_indices:
                pairs.append((ctrl_idx, [ctrl_idx]))  # control predicts itself

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pert_idx, ctrl_indices = self.pairs[idx]

        # Get expression of perturbed cell
        pert_expr = self.adata.X[pert_idx]
        if hasattr(pert_expr, "toarray"):
            pert_expr = pert_expr.toarray().flatten()
        else:
            pert_expr = np.array(pert_expr).flatten()

        # Get expression of ONE representative control (average of matched controls)
        ctrl_exprs = []
        for ci in ctrl_indices:
            e = self.adata.X[ci]
            if hasattr(e, "toarray"):
                e = e.toarray().flatten()
            else:
                e = np.array(e).flatten()
            ctrl_exprs.append(e)

        if len(ctrl_exprs) == 0:
            # Fallback: use the perturbed cell's expression as control (identity mapping)
            ctrl_expr = pert_expr.copy()
        else:
            ctrl_expr = np.mean(ctrl_exprs, axis=0)

        # Select gene subset
        gene_idx = [self.gene_to_idx[g] for g in self.gene_names]
        pert_expr = pert_expr[gene_idx].astype(np.float32)
        ctrl_expr = ctrl_expr[gene_idx].astype(np.float32)

        pert_name = self.adata.obs[self.perturbation_col].iloc[pert_idx]
        pert_idx_enc = self.perturbation_to_idx.get(pert_name, 0)

        return {
            "control": torch.tensor(ctrl_expr, dtype=torch.float32),
            "perturbed": torch.tensor(pert_expr, dtype=torch.float32),
            "perturbation_idx": torch.tensor(pert_idx_enc, dtype=torch.long),
            "perturbation_name": pert_name,
            "control_indices": ctrl_indices,
        }


def collate_paired_batch(batch):
    """Collate a batch of (control, perturbed, perturbation) pairs."""
    controls = torch.stack([b["control"] for b in batch])
    perturbed = torch.stack([b["perturbed"] for b in batch])
    pert_idx = torch.stack([b["perturbation_idx"] for b in batch])
    pert_names = [b["perturbation_name"] for b in batch]

    # Log1p normalize (standard for scRNA-seq)
    controls = torch.log1p(controls)
    perturbed = torch.log1p(perturbed)

    return {
        "control": controls,
        "perturbed": perturbed,
        "perturbation_idx": pert_idx,
        "perturbation_name": pert_names,
    }


def create_train_val_loaders(
    adata,
    perturbation_col: str = "perturbation_name",
    gemgroup_col: str = "gemgroup",
    max_genes: int = 2000,
    train_frac: float = 0.8,
    batch_size: int = 64,
    num_workers: int = 4,
    sample_controls: int = 5,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Create paired train/val DataLoaders.

    Splits by perturbation (not by cell) to ensure OOD evaluation:
    - Train: 80% of perturbations
    - Val: 20% of perturbations (held out)

    Returns:
        train_loader, val_loader, gene_names
    """
    dataset = PairedPerturbationDataset(
        adata=adata,
        perturbation_col=perturbation_col,
        gemgroup_col=gemgroup_col,
        max_genes=max_genes,
        sample_controls_per_perturbed=sample_controls,
    )

    # Split by perturbation TYPE (not by cell) for OOD evaluation
    # This ensures we evaluate on perturbations not seen during training
    all_perts = list(dataset.perturbation_to_idx.keys())
    n_total = len(all_perts)
    n_train = int(n_total * train_frac)

    self.rng = np.random.default_rng(42)
    perms = self.rng.permutation(n_total)
    train_perts = set([all_perts[i] for i in perms[:n_train]])
    val_perts = set([all_perts[i] for i in perms[n_train:]])

    train_indices = [
        i for i, (_, pert_name) in enumerate(dataset.perturbation_to_idx.items())
        if pert_name in train_perts
    ]
    val_indices = [
        i for i, (_, pert_name) in enumerate(dataset.perturbation_to_idx.items())
        if pert_name in val_perts
    ]

    # Actually, we need to split by perturbation INDEX in the pairs
    # Better approach: track which perturbation each pair belongs to
    pert_names_in_pairs = [dataset.adata.obs[dataset.perturbation_col].iloc[p[0]] for p in dataset.pairs]
    all_pert_pairs = list(set(pert_names_in_pairs))
    n_perts = len(all_pert_pairs)
    n_train_perts = int(n_perts * train_frac)

    rng2 = np.random.default_rng(42)
    pert_perms = rng2.permutation(n_perts)
    train_pert_set = set([all_pert_pairs[i] for i in pert_perms[:n_train_perts]])
    val_pert_set = set([all_pert_pairs[i] for i in pert_perms[n_train_perts:]])

    train_idx = [i for i, p in enumerate(pert_names_in_pairs) if p in train_pert_set]
    val_idx = [i for i, p in enumerate(pert_names_in_pairs) if p in val_pert_set]

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_paired_batch,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_paired_batch,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, dataset.gene_names
