from pathlib import Path
import numpy as np
import scanpy as sc

RAW_PATH = Path("data/raw/naive_cd4_t_cells.h5ad")
OUT_PATH = Path("data/interim/naive_cd4_t_cells_downsampled.h5ad")

LABEL_NAME = "Naive_CD4"
DOWNSAMPLE_N = 3000
RANDOM_SEED = 42


def main():
    print("Starting inspection...", flush=True)
    print(f"Raw path: {RAW_PATH.resolve()}", flush=True)
    print(f"File exists: {RAW_PATH.exists()}", flush=True)

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Could not find file: {RAW_PATH}")

    print("\nLoading dataset...", flush=True)
    adata = sc.read_h5ad(RAW_PATH)

    print("\n=== BASIC INFO ===", flush=True)
    print(adata, flush=True)
    print(f"Shape: {adata.shape}", flush=True)
    print(f"Number of cells: {adata.n_obs}", flush=True)
    print(f"Number of genes: {adata.n_vars}", flush=True)

    print("\n=== OBS COLUMNS ===", flush=True)
    print(list(adata.obs.columns), flush=True)

    print("\n=== VAR COLUMNS ===", flush=True)
    print(list(adata.var.columns), flush=True)

    if "cell_type" in adata.obs.columns:
        print("\n=== CELL TYPE COUNTS ===", flush=True)
        print(adata.obs["cell_type"].value_counts().head(30), flush=True)
        print(f"Unique cell types: {adata.obs['cell_type'].nunique()}", flush=True)
        print(f"Missing cell_type: {adata.obs['cell_type'].isna().sum()}", flush=True)
    else:
        print("\nNo 'cell_type' column found in obs.", flush=True)

    print("\n=== MATRIX INFO ===", flush=True)
    print(f"X type: {type(adata.X)}", flush=True)

    try:
        print("First 5x5 of X:", flush=True)
        print(adata.X[:5, :5], flush=True)
    except Exception as e:
        print(f"Could not print X slice: {e}", flush=True)

    print("\n=== RAW / OBSM / UNS ===", flush=True)
    print(f"adata.raw: {adata.raw}", flush=True)
    print(f"obsm keys: {list(adata.obsm.keys())}", flush=True)
    print(f"uns keys: {list(adata.uns.keys())}", flush=True)

    print("\n=== DOWNSAMPLING ===", flush=True)
    n_available = adata.n_obs
    n_take = min(DOWNSAMPLE_N, n_available)
    print(f"Requested cells: {DOWNSAMPLE_N}", flush=True)
    print(f"Available cells: {n_available}", flush=True)
    print(f"Taking cells: {n_take}", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    selected_idx = rng.choice(n_available, size=n_take, replace=False)

    adata_small = adata[selected_idx].copy()
    adata_small.obs["broad_label"] = LABEL_NAME

    print("\n=== DOWNSAMPLED DATASET ===", flush=True)
    print(adata_small, flush=True)
    print(f"Downsampled shape: {adata_small.shape}", flush=True)
    print(adata_small.obs["broad_label"].value_counts(), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata_small.write(OUT_PATH)

    print(f"\nSaved downsampled file to: {OUT_PATH.resolve()}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()