from pathlib import Path
import scanpy as sc
import anndata as ad


DATA_DIR = Path("data/interim")
OUT_PATH = Path("data/processed/immune_4class_merged.h5ad")

FILES = {
    "Naive_CD4": DATA_DIR / "naive_cd4_t_cells_downsampled.h5ad",
    "CD8_related_T": DATA_DIR / "CD8_related_t_cells_downsampled.h5ad",
    "NK_ILC": DATA_DIR / "NK_ILC_downsampled.h5ad",
    "B_plasma": DATA_DIR / "B_plasma_downsampled.h5ad",
}


def load_and_label(path: Path, broad_label: str) -> sc.AnnData:
    print(f"Loading {broad_label}: {path}", flush=True)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    adata = sc.read_h5ad(path)

    adata.obs["broad_label"] = broad_label
    adata.obs["source_subset"] = broad_label

    print(f"  shape: {adata.shape}", flush=True)
    print(f"  broad_label counts:\n{adata.obs['broad_label'].value_counts()}", flush=True)
    return adata


def main():
    adatas = []
    for broad_label, path in FILES.items():
        adata = load_and_label(path, broad_label)
        adatas.append(adata)

    print("\nChecking shared genes...", flush=True)
    shared_genes = set(adatas[0].var_names)
    for adata in adatas[1:]:
        shared_genes &= set(adata.var_names)

    print(f"Shared genes across all files: {len(shared_genes)}", flush=True)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes found across the four files.")

    print("\nMerging with inner join on shared genes...", flush=True)
    merged = ad.concat(
        adatas,
        join="inner",
        merge="same",
        label="concat_source",
        keys=list(FILES.keys()),
        index_unique="-"
    )

    print("\nMerged dataset summary:", flush=True)
    print(merged, flush=True)
    print("\nBroad label counts:", flush=True)
    print(merged.obs["broad_label"].value_counts(), flush=True)

    print("\nMerged obs columns:", flush=True)
    print(list(merged.obs.columns), flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.write(OUT_PATH)

    print(f"\nSaved merged dataset to: {OUT_PATH.resolve()}", flush=True)


if __name__ == "__main__":
    main()