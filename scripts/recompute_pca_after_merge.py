from pathlib import Path

import scanpy as sc


INPUT_PATH = Path("data/processed/immune_4class_split.h5ad")
OUTPUT_PATH = Path("data/processed/immune_4class_split_recomputed_pca.h5ad")

N_TOP_GENES = 3000
N_PCS = 50


def main():
    print("Loading merged split dataset...", flush=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

    adata = sc.read_h5ad(INPUT_PATH)

    print("\nOriginal data:", flush=True)
    print(adata, flush=True)

    print("\nBroad label counts:", flush=True)
    print(adata.obs["broad_label"].value_counts(), flush=True)

    print("\nSelecting highly variable genes...", flush=True)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
    )

    print("Number of highly variable genes:", flush=True)
    print(adata.var["highly_variable"].sum(), flush=True)

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    print("\nHVG subset:", flush=True)
    print(adata_hvg, flush=True)

    print("\nScaling data...", flush=True)
    sc.pp.scale(adata_hvg, max_value=10)

    print("\nComputing PCA...", flush=True)
    sc.tl.pca(
        adata_hvg,
        n_comps=N_PCS,
        svd_solver="arpack",
    )

    print("\nNew obsm keys:", flush=True)
    print(list(adata_hvg.obsm.keys()), flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    adata_hvg.write(OUTPUT_PATH)

    print(f"\nSaved to: {OUTPUT_PATH.resolve()}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()