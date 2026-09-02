from pathlib import Path

import joblib
import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.sparsefuncs import mean_variance_axis


INPUT_PATH = Path("data/processed/immune_finegrained_split.h5ad")
OUTPUT_PATH = Path("data/processed/immune_finegrained_split_trainfit_pca.h5ad")

ARTIFACT_DIR = Path("results/models/preprocessing")
HVG_PATH = ARTIFACT_DIR / "train_only_hvg_genes.txt"
SCALER_PATH = ARTIFACT_DIR / "train_only_scaler.joblib"
PCA_PATH = ARTIFACT_DIR / "train_only_pca.joblib"

SPLIT_COL = "split"
LABEL_COL = "fine_label"

N_TOP_GENES = 3000
N_PCS = 50

PCA_KEY = "X_pca_trainfit"


def to_dense_float32(X):
    """
    Convert sparse or dense matrix to dense float32.
    This is okay for your current project size after selecting HVGs.
    """
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def select_top_variable_genes_train_only(adata, train_mask, n_top_genes):
    """
    Select top variable genes using only training cells.
    This avoids feature-selection leakage.
    """
    print("Selecting highly variable genes using training cells only...", flush=True)

    X_train = adata.X[train_mask, :]

    if sparse.issparse(X_train):
        means, variances = mean_variance_axis(X_train, axis=0)
    else:
        variances = np.var(X_train, axis=0)

    variances = np.asarray(variances).ravel()

    n_top = min(n_top_genes, adata.n_vars)
    top_gene_indices = np.argsort(variances)[-n_top:]

    # Sort indices so gene order is stable
    top_gene_indices = np.sort(top_gene_indices)

    hvg_gene_names = adata.var_names[top_gene_indices].to_numpy()

    print(f"Selected {len(hvg_gene_names)} genes.", flush=True)

    return top_gene_indices, hvg_gene_names


def main():
    print("Loading split dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    adata = sc.read_h5ad(INPUT_PATH)

    print("\n=== DATASET INFO ===", flush=True)
    print(adata, flush=True)

    if SPLIT_COL not in adata.obs.columns:
        raise ValueError(f"Missing split column: {SPLIT_COL}")

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    print("\nSplit counts:", flush=True)
    print(adata.obs[SPLIT_COL].value_counts(), flush=True)

    print("\nLabel counts:", flush=True)
    print(adata.obs[LABEL_COL].value_counts(), flush=True)

    train_mask = adata.obs[SPLIT_COL].values == "train"
    val_mask = adata.obs[SPLIT_COL].values == "val"
    test_mask = adata.obs[SPLIT_COL].values == "test"

    print("\nTrain cells:", train_mask.sum(), flush=True)
    print("Val cells:", val_mask.sum(), flush=True)
    print("Test cells:", test_mask.sum(), flush=True)

    # 1. Select HVGs using training cells only
    top_gene_indices, hvg_gene_names = select_top_variable_genes_train_only(
        adata=adata,
        train_mask=train_mask,
        n_top_genes=N_TOP_GENES,
    )

    # 2. Subset all splits to the same training-selected genes
    print("\nSubsetting to train-selected genes...", flush=True)

    X_train = adata.X[train_mask, :][:, top_gene_indices]
    X_val = adata.X[val_mask, :][:, top_gene_indices]
    X_test = adata.X[test_mask, :][:, top_gene_indices]

    print("Train feature shape before PCA:", X_train.shape, flush=True)
    print("Val feature shape before PCA:", X_val.shape, flush=True)
    print("Test feature shape before PCA:", X_test.shape, flush=True)

    # 3. Convert to dense after gene selection
    print("\nConverting selected matrices to dense float32...", flush=True)

    X_train_dense = to_dense_float32(X_train)
    X_val_dense = to_dense_float32(X_val)
    X_test_dense = to_dense_float32(X_test)

    # 4. Fit scaler on training only, transform all splits
    print("\nFitting scaler on training set only...", flush=True)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_val_scaled = scaler.transform(X_val_dense)
    X_test_scaled = scaler.transform(X_test_dense)

    # 5. Fit PCA on training only, transform all splits
    print("\nFitting PCA on training set only...", flush=True)

    pca = PCA(n_components=N_PCS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("\nExplained variance ratio sum:", flush=True)
    print(float(np.sum(pca.explained_variance_ratio_)), flush=True)

    print("\nPCA shapes:", flush=True)
    print("Train PCA:", X_train_pca.shape, flush=True)
    print("Val PCA:", X_val_pca.shape, flush=True)
    print("Test PCA:", X_test_pca.shape, flush=True)

    # 6. Reassemble PCA matrix in original cell order
    print("\nReassembling PCA features in original cell order...", flush=True)

    X_pca_all = np.zeros((adata.n_obs, N_PCS), dtype=np.float32)
    X_pca_all[train_mask, :] = X_train_pca
    X_pca_all[val_mask, :] = X_val_pca
    X_pca_all[test_mask, :] = X_test_pca

    adata.obsm[PCA_KEY] = X_pca_all

    # Store useful metadata
    adata.uns["train_only_pca"] = {
        "pca_key": PCA_KEY,
        "n_top_genes": int(N_TOP_GENES),
        "n_pcs": int(N_PCS),
        "hvg_selection": "top variance genes selected using training cells only",
        "scaler": "StandardScaler fit on training cells only",
        "pca": "PCA fit on training cells only",
    }

    # 7. Save outputs
    print("\nSaving outputs...", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    adata.write(OUTPUT_PATH)

    with open(HVG_PATH, "w") as f:
        for gene in hvg_gene_names:
            f.write(f"{gene}\n")

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)

    print(f"Saved processed dataset to: {OUTPUT_PATH.resolve()}", flush=True)
    print(f"Saved HVG list to: {HVG_PATH.resolve()}", flush=True)
    print(f"Saved scaler to: {SCALER_PATH.resolve()}", flush=True)
    print(f"Saved PCA model to: {PCA_PATH.resolve()}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()