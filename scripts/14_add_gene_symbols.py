from pathlib import Path

import pandas as pd
import scanpy as sc


ADATA_PATH = Path("data/processed/immune_finegrained_split.h5ad")

TOP_GENES_PATH = Path("results/sparse_logreg/sparse_logreg_top_genes_by_class.csv")
OUT_PATH = Path("results/sparse_logreg/sparse_logreg_top_genes_by_class_with_symbols.csv")


POSSIBLE_SYMBOL_COLUMNS = [
    "gene_symbols",
    "feature_name",
    "gene_name",
    "symbol",
    "gene_symbol",
]


def main():
    print("Loading AnnData file...", flush=True)
    adata = sc.read_h5ad(ADATA_PATH)

    print("\nAvailable var columns:", flush=True)
    print(list(adata.var.columns), flush=True)

    symbol_col = None
    for col in POSSIBLE_SYMBOL_COLUMNS:
        if col in adata.var.columns:
            symbol_col = col
            break

    if symbol_col is None:
        raise ValueError(
            "Could not find a gene symbol column in adata.var. "
            "Check adata.var.columns and update POSSIBLE_SYMBOL_COLUMNS."
        )

    print(f"\nUsing gene symbol column: {symbol_col}", flush=True)

    # Map Ensembl ID / var_name to gene symbol
    gene_id_to_symbol = adata.var[symbol_col].to_dict()

    print("Loading sparse logistic regression top genes...", flush=True)
    top_genes = pd.read_csv(TOP_GENES_PATH)

    if "gene" not in top_genes.columns:
        raise ValueError("Expected a column named 'gene' in the top genes CSV.")

    top_genes["gene_id"] = top_genes["gene"]
    top_genes["gene_symbol"] = top_genes["gene_id"].map(gene_id_to_symbol)

    # Reorder columns to make it easier to read
    first_cols = ["class", "direction", "rank", "gene_id", "gene_symbol", "coefficient"]
    other_cols = [c for c in top_genes.columns if c not in first_cols and c != "gene"]

    top_genes = top_genes[first_cols + other_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_genes.to_csv(OUT_PATH, index=False)

    print(f"\nSaved annotated top genes to: {OUT_PATH.resolve()}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()