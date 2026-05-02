from pathlib import Path
import scanpy as sc

DATA_PATH = Path("data/raw/all_cells.h5ad")

print("Starting script...")
print(f"Looking for file at: {DATA_PATH.resolve()}")
print(f"File exists: {DATA_PATH.exists()}")

print("About to open h5ad in backed mode...")
adata = sc.read_h5ad(DATA_PATH, backed="r")

print("Loaded file.")
print("AnnData object:")
print(adata)

print("\nobs columns:")
print(list(adata.obs.columns))

print("\nvar columns:")
print(list(adata.var.columns))

if "cell_type" in adata.obs.columns:
    print("\nTop cell_type counts:")
    print(adata.obs["cell_type"].value_counts().head(30))

print("Done.")